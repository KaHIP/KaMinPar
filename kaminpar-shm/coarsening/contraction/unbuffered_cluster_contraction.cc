/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.cc
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h"

#include <array>
#include <cstring>
#include <optional>
#include <utility>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/fixed_size_sparse_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {

namespace {

SET_DEBUG(false);

class NeighborhoodsBuffer {
  static constexpr NodeID kMaxNumNodes = 4096;
  static constexpr EdgeID kMaxNumEdges = 32768;

public:
  [[nodiscard]] static bool exceeds_capacity(const NodeID degree) {
    return degree >= kMaxNumEdges;
  }

  NeighborhoodsBuffer(
      EdgeID *nodes,
      NodeID *edges,
      NodeWeight *node_weights,
      EdgeWeight *edge_weights,
      CompactStaticArray<NodeID> &remapping
  )
      : _nodes(nodes),
        _edges(edges),
        _node_weights(node_weights),
        _edge_weights(edge_weights),
        _remapping(remapping),
        _num_buffered_nodes(0),
        _num_buffered_edges(0) {
    _node_buffer[0] = 0;
  }

  [[nodiscard]] NodeID num_buffered_nodes() const {
    return _num_buffered_nodes;
  }

  [[nodiscard]] NodeID num_buffered_edges() const {
    return _num_buffered_edges;
  }

  [[nodiscard]] bool overfills(const NodeID degree) const {
    return _num_buffered_nodes + 1 >= kMaxNumNodes || _num_buffered_edges + degree >= kMaxNumEdges;
  }

  template <typename Lambda>
  void add(const NodeID c_u, const NodeID degree, const NodeWeight weight, Lambda &&it) {
    _remapping_buffer[_num_buffered_nodes] = c_u;
    _node_buffer[_num_buffered_nodes + 1] = degree + _node_buffer[_num_buffered_nodes];
    _node_weight_buffer[_num_buffered_nodes] = weight;
    _num_buffered_nodes += 1;

    it([this](const auto c_v, const auto weight) {
      _edge_buffer[_num_buffered_edges] = c_v;
      _edge_weight_buffer[_num_buffered_edges] = weight;
      _num_buffered_edges += 1;
    });
  }

  void flush(const NodeID c_u, const EdgeID e) {
    const std::size_t buffered_edge_weights_size = _num_buffered_edges * sizeof(EdgeWeight);
    std::memcpy(_edge_weights + e, _edge_weight_buffer.data(), buffered_edge_weights_size);

    const std::size_t buffered_edges_size = _num_buffered_edges * sizeof(NodeID);
    std::memcpy(_edges + e, _edge_buffer.data(), buffered_edges_size);

    const std::size_t buffered_node_weights_size = _num_buffered_nodes * sizeof(NodeWeight);
    std::memcpy(_node_weights + c_u, _node_weight_buffer.data(), buffered_node_weights_size);

    for (NodeID i = 0; i < _num_buffered_nodes; ++i) {
      _remapping.write(_remapping_buffer[i], c_u + i);
      _nodes[c_u + i] = _node_buffer[i] + e;
    }

    _num_buffered_nodes = 0;
    _num_buffered_edges = 0;
    _node_buffer[0] = 0;
  }

private:
  EdgeID *_nodes;
  NodeID *_edges;
  NodeWeight *_node_weights;
  EdgeWeight *_edge_weights;
  CompactStaticArray<NodeID> &_remapping;

  NodeID _num_buffered_nodes;
  EdgeID _num_buffered_edges;
  std::array<NodeID, kMaxNumNodes> _remapping_buffer;
  std::array<EdgeID, kMaxNumNodes> _node_buffer;
  std::array<NodeWeight, kMaxNumNodes> _node_weight_buffer;
  std::array<NodeID, kMaxNumEdges> _edge_buffer;
  std::array<EdgeWeight, kMaxNumEdges> _edge_weight_buffer;
};

template <typename Graph>
std::unique_ptr<CoarseGraph> contract_clustering_unbuffered(
    const Graph &graph,
    const NodeID c_n,
    StaticArray<NodeID> mapping,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  const auto &buckets_index = m_ctx.buckets_index;
  const auto &buckets = m_ctx.buckets;

  // To contract the graph, we iterate over the coarse nodes in parallel and aggregate the
  // neighborhood of the coarse nodes. The neighborhoods are aggregated by iterating over the
  // neighborhoods of the fine nodes that are contracted to a coarse node. After aggregating the
  // neighborhood of a coarse node, the neighborhood is directly written to the coarse edge array
  // and the offset is stored in the coarse node array. The correct offset into the coarse edge
  // array is obtained by atomically incrementing a 128-bit integer that contains the next coarse
  // node ID (for remapping) and coarse edge offset. For that to work, we remap the coarse nodes
  // afterwards.
  //
  // To improve performance, we temporarily store small neighborhoods of coarse nodes in a
  // thread-local constant-sized buffer, which is copied to the coarse edge array when it becomes
  // full. In doing so, multiple atomic increments of the 128-bit counter are combined into a single
  // incrementation.
  //
  // To aggregate the coarse neighborhoods, we provide three implementations with different memory
  // consumption characteristics: fixed-size large rating maps, growing hash tables, and a two-phase
  // approach.

  // Allocate the coarse node (weight) and edge (weight) array. We overcomit memory for the edge and
  // edge weight array as we only know the amount of edges of the coarse graph afterwards.
  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1, static_array::noinit);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n, static_array::noinit);
  STOP_HEAP_PROFILER();

  const EdgeID num_fine_edges = graph.m();
  auto c_edges_ptr = heap_profiler::overcommit_memory<NodeID>(num_fine_edges);
  auto c_edge_weights_ptr = heap_profiler::overcommit_memory<EdgeWeight>(num_fine_edges);
  NodeID *c_edges = c_edges_ptr.get();
  EdgeWeight *c_edge_weights = c_edge_weights_ptr.get();
  STOP_TIMER();

  START_HEAP_PROFILER("Construct coarse graph");
  START_TIMER("Construct coarse graph");

  // When writing the neighborhood of a coarse node into the coarse node array, we additionally
  // store inside a vector the coarse node ID it is remapped to. This allows us to directly write
  // the coarse edges into the coarse edge array.
  CompactStaticArray<NodeID> remapping(math::byte_width(c_n), c_n);
  const auto write_neighbourhood = [&](const NodeID c_u,
                                       const NodeWeight c_u_weight,
                                       const NodeID new_c_u,
                                       EdgeID edge,
                                       auto &map) {
    remapping.write(c_u, new_c_u);

    c_nodes[new_c_u] = edge;
    c_node_weights[new_c_u] = c_u_weight;

    for (const auto [c_v, weight] : map.entries()) {
      c_edges[edge] = c_v;
      c_edge_weights[edge] = weight;
      edge += 1;
    }
  };

  // We keep track of the new coarse node ID for the next coarse node to be remapped and its edge
  // offset inside a 128-bit integer. We atomically increment it with double-width compare-and-swap
  // instructions.
  union alignas(16) NextCoarseNodeInfo {
    __uint128_t val;
    struct {
      std::uint64_t c_u;
      std::uint64_t edge;
    } info;
  };

  NextCoarseNodeInfo next_coarse_node_info = {.info = {.c_u = 0, .edge = 0}};
  const auto atomic_fetch_next_coarse_node_info = [&](const std::uint64_t nodes,
                                                      const std::uint64_t degree) {
    NextCoarseNodeInfo expected;

    bool success;
    do {
      expected = next_coarse_node_info;
      const NextCoarseNodeInfo desired = {
          .info = {.c_u = expected.info.c_u + nodes, .edge = expected.info.edge + degree}
      };
      success = __sync_bool_compare_and_swap(&next_coarse_node_info.val, expected.val, desired.val);
    } while (!success);

    return std::make_pair(expected.info.c_u, expected.info.edge);
  };

  // After aggregating the neighborhood of a coarse node, we want to write the edges into the coarse
  // edge array. However, to reduce the amount of double-width compare-and-swap instructions and
  // therefore to improve the running time, we buffer the coarse edges and only write directly to
  // the coarse edge array if a) the buffer is only capable of containing the single coarse
  // neighborhood that has been aggregated b) the buffer would become full if the coarse
  // neighborhood is added.
  const auto transfer_edges = [&](const NodeID c_u,
                                  const NodeWeight c_u_weight,
                                  auto &edge_collector,
                                  auto &neighborhood_buffer) {
    const std::size_t degree = edge_collector.size();

    if (NeighborhoodsBuffer::exceeds_capacity(degree)) {
      auto [new_c_u, edge] = atomic_fetch_next_coarse_node_info(1, degree);
      write_neighbourhood(c_u, c_u_weight, new_c_u, edge, edge_collector);
      return;
    }

    if (neighborhood_buffer.overfills(degree)) {
      const NodeID num_buffered_nodes = neighborhood_buffer.num_buffered_nodes();
      const EdgeID num_buffered_edges = neighborhood_buffer.num_buffered_edges();

      const auto [new_c_u, edge] =
          atomic_fetch_next_coarse_node_info(num_buffered_nodes + 1, num_buffered_edges + degree);

      neighborhood_buffer.flush(new_c_u, edge);
      write_neighbourhood(
          c_u, c_u_weight, new_c_u + num_buffered_nodes, edge + num_buffered_edges, edge_collector
      );
      return;
    }

    neighborhood_buffer.add(c_u, degree, c_u_weight, [&](auto &&l) {
      for (const auto [c_v, weight] : edge_collector.entries()) {
        l(c_v, weight);
      }
    });
  };

  // After aggregating all coarse neighborhoods, we have to flush the thread-local buffers which
  // might contain remaining coarse neighborhoods.
  const auto flush_edges = [&](auto &neighborhood_buffer) {
    const NodeID num_buffered_nodes = neighborhood_buffer.num_buffered_nodes();
    if (num_buffered_nodes == 0) {
      return;
    }

    const EdgeID num_buffered_edges = neighborhood_buffer.num_buffered_edges();
    auto [new_c_u, edge] =
        atomic_fetch_next_coarse_node_info(num_buffered_nodes, num_buffered_edges);
    neighborhood_buffer.flush(new_c_u, edge);
  };

  // To aggregate the edges of a coarse node for the growing hash tables and single-phase
  // implementation, we simply iterate over the fine edges of the nodes that are contracted to the
  // coarse node and meanwhile compute the coarse node weight.
  const auto aggregate_edges = [&](const NodeID c_u,
                                   const NodeID bucket_start,
                                   const NodeID bucket_end,
                                   auto &local_edge_collector) {
    NodeWeight c_u_weight = 0;

    for (NodeID i = bucket_start; i < bucket_end; ++i) {
      const NodeID u = buckets[i];
      c_u_weight += graph.node_weight(u);

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const NodeID c_v = mapping[v];
        if (c_u != c_v) {
          local_edge_collector[c_v] += w;
        }
      });
    }

    return c_u_weight;
  };

  tbb::enumerable_thread_specific<NeighborhoodsBuffer> neighborhoods_buffer_ets{[&] {
    return NeighborhoodsBuffer(
        c_nodes.data(), c_edges, c_node_weights.data(), c_edge_weights, remapping
    );
  }};

  START_TIMER("Aggregate coarse edges");
  if (con_ctx.unbuffered_implementation == ContractionImplementation::GROWING_HASH_TABLES) {
    using EdgeCollector = DynamicRememberingFlatMap<NodeID, EdgeWeight, NodeID>;
    tbb::enumerable_thread_specific<EdgeCollector> edge_collector_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_edge_collector = edge_collector_ets.local();
      auto &local_buffer = neighborhoods_buffer_ets.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        const NodeWeight c_u_weight = aggregate_edges(c_u, first, last, local_edge_collector);
        transfer_edges(c_u, c_u_weight, local_edge_collector, local_buffer);
        local_edge_collector.clear();
      }
    });

    tbb::parallel_for(neighborhoods_buffer_ets.range(), [&](const auto &local_buffers) {
      for (auto &local_buffer : local_buffers) {
        flush_edges(local_buffer);
      }
    });
  } else if (con_ctx.unbuffered_implementation == ContractionImplementation::SINGLE_PHASE) {
    using EdgeCollector = RatingMap<EdgeWeight, NodeID>;
    tbb::enumerable_thread_specific<EdgeCollector> edge_collector_ets{[&] {
      return EdgeCollector(c_n);
    }};

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_edge_collector = edge_collector_ets.local();
      auto &local_buffer = neighborhoods_buffer_ets.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        // To select the right map, we compute a upper bound on the coarse node degree by summing
        // the degree of all fine nodes.
        NodeID upper_bound_degree = 0;
        for (NodeID i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          upper_bound_degree += graph.degree(u);
        }

        local_edge_collector.execute(upper_bound_degree, [&](auto &edge_collector) {
          const NodeWeight c_u_weight = aggregate_edges(c_u, first, last, edge_collector);
          transfer_edges(c_u, c_u_weight, edge_collector, local_buffer);
          edge_collector.clear();
        });
      }
    });

    tbb::parallel_for(neighborhoods_buffer_ets.range(), [&](const auto &local_buffers) {
      for (auto &local_buffer : local_buffers) {
        flush_edges(local_buffer);
      }
    });
  } else if (con_ctx.unbuffered_implementation == ContractionImplementation::TWO_PHASE) {
    using EdgeCollector = FixedSizeSparseMap<NodeID, EdgeWeight, 65536>;
    tbb::enumerable_thread_specific<EdgeCollector> edge_collector_ets;

    // To aggregate the edges of a coarse node for the two-phase implementation, we iterate over the
    // fine edges of the nodes that are contracted to the coarse node and bump the coarse node to
    // the second phase if its degree crosses a threshold.
    constexpr NodeID kDegreeThreshold = EdgeCollector::MAP_SIZE / 3;
    const auto aggregate_edges = [&](const NodeID c_u,
                                     auto &local_edge_collector) -> std::optional<NodeWeight> {
      const NodeID bucket_start = buckets_index[c_u];
      const NodeID bucket_end = buckets_index[c_u + 1];

      NodeWeight c_u_weight = 0;
      bool second_phase_node = false;
      for (NodeID i = bucket_start; i < bucket_end; ++i) {
        const NodeID u = buckets[i];
        c_u_weight += graph.node_weight(u);

        graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          const NodeID c_v = mapping[v];
          if (c_u != c_v) {
            local_edge_collector[c_v] += w;

            if (local_edge_collector.size() >= kDegreeThreshold) [[unlikely]] {
              second_phase_node = true;
              return true;
            }
          }
          return false;
        });

        if (second_phase_node) [[unlikely]] {
          return std::nullopt;
        }
      }

      return c_u_weight;
    };

    START_TIMER("First phase");
    tbb::concurrent_vector<NodeID> second_phase_nodes;
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_edge_collector = edge_collector_ets.local();
      auto &local_buffer = neighborhoods_buffer_ets.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        if (const auto c_u_weight = aggregate_edges(c_u, local_edge_collector)) {
          transfer_edges(c_u, *c_u_weight, local_edge_collector, local_buffer);
        } else {
          second_phase_nodes.push_back(c_u);
        }

        local_edge_collector.clear();
      }
    });

    tbb::parallel_for(neighborhoods_buffer_ets.range(), [&](const auto &local_buffers) {
      for (auto &local_buffer : local_buffers) {
        flush_edges(local_buffer);
      }
    });
    STOP_TIMER();

    START_TIMER("Second phase");
    ConcurrentFastResetArray<EdgeWeight, NodeID> edge_collector(c_n);
    tbb::enumerable_thread_specific<NodeWeight> coarse_node_weight_ets(0);

    auto &[cur_c_u, cur_edge] = next_coarse_node_info.info;
    for (const NodeID c_u : second_phase_nodes) {
      const NodeID bucket_start = buckets_index[c_u];
      const NodeID bucket_end = buckets_index[c_u + 1];

      // To avoid possible contention caused by atomic fetch-and-add operations when there are
      // clusters in the neighborhood of a coarse node that are often encountered, we first store
      // the edge weights to adjacent coarse nodes in the thread-local hash tables of the first
      // phase and flush them when they are full.
      const auto flush_local_rating_map = [&](auto &local_used_entries,
                                              auto &local_edge_collector) {
        for (const auto [c_v, w] : local_edge_collector.entries()) {
          const EdgeWeight prev_weight =
              __atomic_fetch_add(&edge_collector[c_v], w, __ATOMIC_RELAXED);

          if (prev_weight == 0) {
            local_used_entries.push_back(c_v);
          }
        }

        local_edge_collector.clear();
      };

      // Since there can be many fine nodes with a low degree inside a cluster, parallel iteration
      // only over the neighborhood of fine nodes can lead to inferior running times. For this
      // reason, we iterate in parallel over the fine nodes in a cluster. Furthermore, we iterate in
      // parallel (with nested parallelism) over the neighborhood of fine nodes, if they have a high
      // degree, to reduce load imbalance.
      constexpr NodeID kParallelIterationGrainsize = 2000;
      tbb::parallel_for(tbb::blocked_range<NodeID>(bucket_start, bucket_end), [&](const auto &r) {
        auto &local_coarse_node_weight = coarse_node_weight_ets.local();
        auto &local_used_entries = edge_collector.local_used_entries();
        auto &local_edge_collector = edge_collector_ets.local();

        const NodeID local_bucket_start = r.begin();
        const NodeID local_bucket_end = r.end();
        for (NodeID i = local_bucket_start; i < local_bucket_end; ++i) {
          const NodeID u = buckets[i];
          local_coarse_node_weight += graph.node_weight(u);

          if (graph.degree(u) < kParallelIterationGrainsize * 2) {
            graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
              const NodeID c_v = mapping[v];
              if (c_u != c_v) {
                local_edge_collector[c_v] += w;

                if (local_edge_collector.size() >= kDegreeThreshold) [[unlikely]] {
                  flush_local_rating_map(local_used_entries, local_edge_collector);
                }
              }
            });

            continue;
          }

          graph.pfor_adjacent_nodes(
              u,
              std::numeric_limits<NodeID>::max(),
              kParallelIterationGrainsize,
              [&](auto &&pfor_adjacent_nodes) {
                auto &local_used_entries = edge_collector.local_used_entries();
                auto &local_edge_collector = edge_collector_ets.local();

                pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
                  const NodeID c_v = mapping[v];
                  if (c_u != c_v) {
                    local_edge_collector[c_v] += w;

                    if (local_edge_collector.size() >= kDegreeThreshold) [[unlikely]] {
                      flush_local_rating_map(local_used_entries, local_edge_collector);
                    }
                  }
                });
              }
          );
        }
      });

      tbb::parallel_for(edge_collector_ets.range(), [&](const auto &local_edge_collectors) {
        auto &local_used_entries = edge_collector.local_used_entries();
        for (auto &local_edge_collector : local_edge_collectors) {
          flush_local_rating_map(local_used_entries, local_edge_collector);
        }
      });

      remapping.write(c_u, cur_c_u);

      NodeWeight c_u_weight = 0;
      for (auto &local_c_u_weight : coarse_node_weight_ets) {
        c_u_weight += local_c_u_weight;
        local_c_u_weight = 0;
      }

      c_nodes[cur_c_u] = cur_edge;
      c_node_weights[cur_c_u] = c_u_weight;
      cur_c_u += 1;

      edge_collector.iterate_and_reset([&](const auto, const auto &local_neighbors) {
        EdgeID local_cur_edge =
            __atomic_fetch_add(&cur_edge, local_neighbors.size(), __ATOMIC_RELAXED);

        for (const auto [c_v, w] : local_neighbors) {
          c_edges[local_cur_edge] = c_v;
          c_edge_weights[local_cur_edge] = w;
          local_cur_edge += 1;
        }
      });
    }
    STOP_TIMER();

    DBG << "Unbuffered Contraction:";
    DBG << " First Phase:  " << (c_n - second_phase_nodes.size()) << " clusters";
    DBG << " Second Phase: " << second_phase_nodes.size() << " clusters";
  }

  const EdgeID c_m = next_coarse_node_info.info.edge;
  c_nodes[c_n] = c_m;
  STOP_TIMER();

  // After aggregation, we have to remap the adjacent coarse nodes stored in the coarse edge array
  // as well as the mapping which is used for projection during uncoarsening, because we remapped
  // the coarse nodes during aggregation.
  START_TIMER("Remap coarse edges and mapping");
  tbb::parallel_for(tbb::blocked_range<EdgeID>(0, c_m), [&](const auto &r) {
    for (EdgeID e = r.begin(); e != r.end(); ++e) {
      c_edges[e] = remapping[c_edges[e]];
    }
  });

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      mapping.write(u, remapping[mapping[u]]);
    }
  });
  STOP_TIMER();

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Coarse graph edges allocation");
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(c_edges, c_m * sizeof(NodeID));
    heap_profiler::HeapProfiler::global().record_alloc(c_edge_weights, c_m * sizeof(EdgeWeight));
  }

  RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, std::move(c_edges_ptr));
  RECORD("c_edge_weights")
  StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, std::move(c_edge_weights_ptr));
  STOP_HEAP_PROFILER();

  return std::make_unique<CoarseGraphImpl>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(finalized_c_edges),
          std::move(c_node_weights),
          std::move(finalized_c_edge_weights)
      )),
      std::move(mapping)
  );
}

} // namespace

std::unique_ptr<CoarseGraph> contract_clustering_unbuffered(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = compute_mapping(graph, std::move(clustering), m_ctx);
  fill_cluster_buckets(c_n, graph, mapping, m_ctx.buckets_index, m_ctx.buckets);
  return reified(graph, [&](auto &graph) {
    return contract_clustering_unbuffered(graph, c_n, std::move(mapping), con_ctx, m_ctx);
  });
}

} // namespace kaminpar::shm::contraction
