/*******************************************************************************
 * Contracts clusterings and constructs the coarse graph.
 *
 * @file:   cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/graphutils/cluster_contraction.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
using namespace contraction;

template <template <typename> typename Mapping> class CoarseGraphImpl : public CoarseGraph {
public:
  CoarseGraphImpl(Graph graph, Mapping<NodeID> mapping)
      : _graph(std::move(graph)),
        _mapping(std::move(mapping)) {}

  const Graph &get() const final {
    return _graph;
  }

  Graph &get() final {
    return _graph;
  }

  void project(const StaticArray<BlockID> &array, StaticArray<BlockID> &onto) final {
    tbb::parallel_for<std::size_t>(0, onto.size(), [&](const std::size_t i) {
      onto[i] = array[_mapping[i]];
    });
  }

private:
  Graph _graph;
  Mapping<NodeID> _mapping;
};

namespace {
template <typename Graph, typename Clustering>
void fill_leader_mapping(
    const Graph &graph,
    const Clustering &clustering,
    scalable_vector<parallel::Atomic<NodeID>> &leader_mapping
) {
  START_TIMER("Allocation");
  if (leader_mapping.size() < graph.n()) {
    leader_mapping.resize(graph.n());
  }
  STOP_TIMER();

  RECORD("leader_mapping");
  RECORD_LOCAL_DATA_STRUCT(
      "scalable_vector<parallel::Atomic<NodeID>>",
      leader_mapping.capacity() * sizeof(parallel::Atomic<NodeID>)
  );

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const NodeID u) {
    leader_mapping[clustering[u]].store(1, std::memory_order_relaxed);
  });
  parallel::prefix_sum(
      leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
  );
  STOP_TIMER();
}

template <typename Graph, typename Clustering>
StaticArray<NodeID> compute_mapping(
    const Graph &graph,
    const Clustering &clustering,
    const scalable_vector<parallel::Atomic<NodeID>> &leader_mapping
) {
  START_TIMER("Allocation");
  RECORD("mapping") StaticArray<NodeID> mapping(graph.n());
  STOP_TIMER();

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]] - 1; });
  STOP_TIMER();

  return mapping;
}

template <typename Graph, typename Clustering>
CompactStaticArray<NodeID> compute_compact_mapping(
    const Graph &graph,
    const Clustering &clustering,
    const scalable_vector<parallel::Atomic<NodeID>> &leader_mapping
) {
  const NodeID c_n = leader_mapping[graph.n() - 1];

  START_TIMER("Allocation");
  RECORD("mapping") CompactStaticArray<NodeID> mapping(math::byte_width(c_n), graph.n());
  STOP_TIMER();

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) { mapping.write(u, leader_mapping[clustering[u]] - 1); });
  STOP_TIMER();

  return mapping;
}

template <typename Graph, typename Mapping>
void fill_cluster_buckets(
    const NodeID c_n,
    const Graph &graph,
    const Mapping &mapping,
    scalable_vector<parallel::Atomic<NodeID>> &buckets_index,
    scalable_vector<NodeID> &buckets
) {
  START_TIMER("Allocation");
  if (buckets.size() < graph.n()) {
    buckets.resize(graph.n());
  }
  buckets_index.clear();
  buckets_index.resize(c_n + 1);
  STOP_TIMER();

  RECORD("buckets");
  RECORD_LOCAL_DATA_STRUCT("scalable_vector<NodeID>", buckets.capacity() * sizeof(NodeID));

  RECORD("buckets_index");
  RECORD_LOCAL_DATA_STRUCT(
      "scalable_vector<parallel::Atomic<NodeID>>",
      buckets_index.capacity() * sizeof(parallel::Atomic<NodeID>)
  );

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) {
    buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed);
  });

  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  KASSERT(buckets_index.back() <= graph.n());

  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });
  STOP_TIMER();
}

template <template <typename> typename Mapping, typename Graph, typename Clustering>
std::pair<NodeID, Mapping<NodeID>>
preprocess(const Graph &graph, Clustering &clustering, MemoryContext &m_ctx) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  fill_leader_mapping(graph, clustering, leader_mapping);
  Mapping<NodeID> mapping;
  if constexpr (std::is_same_v<Mapping<NodeID>, StaticArray<NodeID>>) {
    mapping = compute_mapping(graph, clustering, leader_mapping);
  } else {
    mapping = compute_compact_mapping(graph, clustering, leader_mapping);
  }

  const NodeID c_n = leader_mapping[graph.n() - 1];

  TIMED_SCOPE("Allocation") {
    leader_mapping.clear();
    leader_mapping.shrink_to_fit();
    clustering.clear();
    clustering.shrink_to_fit();
  };

  fill_cluster_buckets(c_n, graph, mapping, buckets_index, buckets);

  return {c_n, std::move(mapping)};
}

template <template <typename> typename Mapping, typename Graph, typename Clustering>
std::unique_ptr<CoarseGraph> contract_with_edgebuffer(
    const Graph &graph,
    Clustering &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = preprocess<Mapping>(graph, clustering, m_ctx);

  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  // We build the coarse graph in multiple iterations. In each iteration we compute for a part of
  // the coarse nodes the degree and node weight of each node. We can't insert the edges and edge
  // weights into the arrays yet, because positioning edges in those arrays depends on c_nodes,
  // which we only have after computing a prefix sum over the coarse node degrees in the current
  // part. Thus, we store the edges and edge weights in a temporary buffer and compute the prefix
  // sum and insert the edges and edge weights into the arrays after processing the part.

  const EdgeID edge_count = graph.m();

  // Split the coarse nodes into chunks, such that the edges of the fine graph are roughly split
  // among the chunks.
  START_TIMER("Compute coarse node chunks");
  std::vector<std::pair<NodeID, NodeID>> cluster_chunks;

  // If there are too few fine edges then don't split the coarse nodes into chunks, as this
  // provides little memory benefits.
  const bool split_coarse_nodes = edge_count >= 100000 && con_ctx.edge_buffer_fill_fraction < 1;
  if (!split_coarse_nodes) {
    cluster_chunks.emplace_back(0, c_n);
  } else {
    // Compute the fine degrees of the coarse nodes in parallel and store them temporarily in the
    // (unused) c_nodes array.
    tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
      const NodeID first = buckets_index[c_u];
      const NodeID last = buckets_index[c_u + 1];

      NodeID fine_degree = 0;
      for (NodeID u = first; u < last; ++u) {
        const NodeID v = buckets[u];
        fine_degree += graph.degree(v);
      }

      c_nodes[c_u + 1] = fine_degree;
    });

    const EdgeID max_chunk_edge_count = edge_count * con_ctx.edge_buffer_fill_fraction;

    NodeID chunk_start = 0;
    NodeID chunk_edge_count = 0;
    for (NodeID c_u = 0; c_u < c_n; ++c_u) {
      const NodeID fine_degree = c_nodes[c_u + 1];
      chunk_edge_count += fine_degree;

      // With this coarse node the chunk would have more fine edges than the maximum allowed
      // limit. Thus, create a new chunk.
      if (chunk_edge_count >= max_chunk_edge_count) {
        // It might happen that the chunk only consists of a high degree node which crosses the
        // limit.
        if (chunk_start == c_u) {
          cluster_chunks.emplace_back(c_u, c_u + 1);
          chunk_start = c_u + 1;
        } else {
          cluster_chunks.emplace_back(chunk_start, c_u);
          chunk_start = c_u;
        }

        chunk_edge_count = 0;
      }
    }

    // Create a chunk for the last coarse nodes, if the last coarse node did not cross the limit.
    if (chunk_start != c_n) {
      cluster_chunks.emplace_back(chunk_start, c_n);
    }
  }

  STOP_TIMER();

  // Overcomit memory for the edge and edge weight array as we only know the amount of edges of
  // the coarse graph afterwards.
  NodeID *c_edges;
  EdgeWeight *c_edge_weights;
  if constexpr (kHeapProfiling) {
    // As we overcommit memory do not track the amount of bytes used directly. Instead record it
    // manually afterwards.
    c_edges = (NodeID *)heap_profiler::std_malloc(edge_count * sizeof(NodeID));
    c_edge_weights = (EdgeWeight *)heap_profiler::std_malloc(edge_count * sizeof(EdgeWeight));
  } else {
    c_edges = (NodeID *)std::malloc(edge_count * sizeof(NodeID));
    c_edge_weights = (EdgeWeight *)std::malloc(edge_count * sizeof(EdgeWeight));
  }

  START_HEAP_PROFILER("Construct coarse graph");
  START_TIMER("Construct coarse graph");

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};
  NavigableLinkedList<NodeID, Edge, scalable_vector> edge_buffer_ets;

  for (const auto [cluster_start, cluster_end] : cluster_chunks) {
    tbb::parallel_for(tbb::blocked_range<NodeID>(cluster_start, cluster_end), [&](const auto &r) {
      auto &local_collector = collector.local();
      auto &local_edge_buffer = edge_buffer_ets.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        local_edge_buffer.mark(c_u);

        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        // Build coarse graph
        const auto collect_edges = [&](auto &map) {
          NodeWeight c_u_weight = 0;
          for (NodeID i = first; i < last; ++i) {
            const NodeID u = buckets[i];
            KASSERT(mapping[u] == c_u);

            c_u_weight += graph.node_weight(u); // coarse node weight

            // collect coarse edges
            graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
              const NodeID c_v = mapping[v];
              if (c_u != c_v) {
                map[c_v] += graph.edge_weight(e);
              }
            });
          }

          c_node_weights[c_u] = c_u_weight; // coarse node weights are done now
          c_nodes[c_u + 1] = map.size();    // node degree (used to build c_nodes)

          // Since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes
          // degree), we can't place the edges of c_u in the c_edges and c_edge_weights arrays.
          // Hence, we store them in auxiliary arrays and note their position in the auxiliary
          // arrays
          for (const auto [c_v, weight] : map.entries()) {
            local_edge_buffer.push_back({c_v, weight});
          }

          map.clear();
        };

        // To select the right map, we need a upper bound on the coarse node degree. If we
        // previously split the coarse nodes into chunks, we have already computed them and stored
        // them in the c_nodes array.
        NodeID upper_bound_degree;
        if (split_coarse_nodes) {
          upper_bound_degree = c_nodes[c_u + 1];
        } else {
          upper_bound_degree = 0;

          for (NodeID i = first; i < last; ++i) {
            const NodeID u = buckets[i];
            upper_bound_degree += graph.degree(u);
          }
        }

        local_collector.execute(upper_bound_degree, collect_edges);
      }
    });

    parallel::prefix_sum(
        c_nodes.begin() + cluster_start,
        c_nodes.begin() + cluster_end + 1,
        c_nodes.begin() + cluster_start
    );

    parallel::Atomic<std::size_t> global_pos = 0;
    std::size_t num_markers = 0;
    for (const auto &local_list : edge_buffer_ets) {
      num_markers += local_list.markers().size();
    }
    if (all_buffered_nodes.size() < num_markers) {
      all_buffered_nodes.resize(num_markers);
    }

    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for(edge_buffer_ets.range(), [&](auto &r) {
            for (auto &local_list : r) {
              local_list.flush();
            }
          });
        },
        [&] {
          tbb::parallel_for(edge_buffer_ets.range(), [&](const auto &r) {
            for (const auto &local_list : r) {
              const auto &markers = local_list.markers();
              const std::size_t local_pos = global_pos.fetch_add(markers.size());
              std::copy(markers.begin(), markers.end(), all_buffered_nodes.begin() + local_pos);
            }
          });
        }
    );

    tbb::parallel_for<NodeID>(cluster_start, cluster_end, [&](const NodeID i) {
      const auto &marker = all_buffered_nodes[i - cluster_start];
      const auto *list = marker.local_list;
      const NodeID c_u = marker.key;

      const NodeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
      const EdgeID first_target_index = c_nodes[c_u];
      const EdgeID first_source_index = marker.position;

      for (EdgeID j = 0; j < c_u_degree; ++j) {
        const auto to = first_target_index + j;
        const auto [c_v, weight] = list->get(first_source_index + j);
        c_edges[to] = c_v;
        c_edge_weights[to] = weight;
      }
    });

    edge_buffer_ets.clear();
    all_buffered_nodes.clear();
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  START_HEAP_PROFILER("Coarse graph edges allocation");
  RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, c_edges);
  RECORD("c_edge_weights") StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, c_edge_weights);
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(c_edges, c_m * sizeof(NodeID));
    heap_profiler::HeapProfiler::global().record_alloc(c_edge_weights, c_m * sizeof(EdgeWeight));
  }
  STOP_HEAP_PROFILER();

  return std::make_unique<CoarseGraphImpl<Mapping>>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(finalized_c_edges),
          std::move(c_node_weights),
          std::move(finalized_c_edge_weights)
      )),
      std::move(mapping)
  );
}

template <template <typename> typename Mapping, typename Graph, typename Clustering>
std::unique_ptr<CoarseGraph> contract_without_edgebuffer_naive(
    const Graph &graph,
    Clustering &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = preprocess<Mapping>(graph, clustering, m_ctx);

  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  //
  // We build the coarse graph in two steps. First, we compute the degree and node weight of each
  // coarse node. We store the degree in the node array and the node weights in the node weight
  // array. We compute a prefix sum over all coarse node degrees to have the correct offsets in
  // the node array. Additionally, we compute the max edge weight. Then, we allocate the edge and
  // edge weight array with compact IDs. In the second step, we compute the edges and edge weights
  // again and store them in the corresponding arrays.
  //

  START_HEAP_PROFILER("Construct coarse nodes");
  START_TIMER("Construct coarse nodes");

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};
  tbb::enumerable_thread_specific<std::size_t> max_edge_weight_ets;

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();

    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      const NodeID first = buckets_index[c_u];
      const NodeID last = buckets_index[c_u + 1];

      // Build coarse graph
      const auto collect_edges = [&](auto &map) {
        NodeWeight c_u_weight = 0;
        for (NodeID i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          KASSERT(mapping[u] == c_u);

          c_u_weight += graph.node_weight(u);

          // Collect coarse edges
          graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += graph.edge_weight(e);
            }
          });
        }

        c_nodes[c_u + 1] = map.size(); // Store node degree which is used to build c_nodes
        c_node_weights[c_u] = c_u_weight;

        std::size_t max_edge_weight = max_edge_weight_ets.local();
        for (const auto [c_v, weight] : map.entries()) {
          max_edge_weight = std::max(max_edge_weight, math::abs(weight));
        }

        max_edge_weight_ets.local() = max_edge_weight;
        map.clear();
      };

      // To select the right map, we compute a upper bound on the coarse node degree by summing
      // the degree of all fine nodes.
      NodeID upper_bound_degree = 0;
      for (NodeID i = first; i < last; ++i) {
        const NodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }

      local_collector.execute(upper_bound_degree, collect_edges);
    }
  });

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

  std::size_t max_edge_weight = 0;
  for (const std::size_t edge_weight : max_edge_weight_ets) {
    max_edge_weight = std::max(max_edge_weight, edge_weight);
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  START_HEAP_PROFILER("Coarse graph edges allocation");
  START_TIMER("Allocation");
  RECORD("c_edges") StaticArray<NodeID> c_edges(c_m, StaticArray<NodeID>::no_init{});
  RECORD("c_edge_weights")
  StaticArray<EdgeWeight> c_edge_weights(c_m, StaticArray<EdgeWeight>::no_init{});
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Construct coarse edges");
  START_TIMER("Construct coarse edges");

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();

    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      const NodeID first = buckets_index[c_u];
      const NodeID last = buckets_index[c_u + 1];

      // Build coarse graph
      const auto collect_edges = [&](auto &map) {
        for (NodeID i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          KASSERT(mapping[u] == c_u);

          // Collect coarse edges
          graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += graph.edge_weight(e);
            }
          });
        }

        EdgeID edge = c_nodes[c_u];
        for (const auto [c_v, weight] : map.entries()) {
          c_edges[edge] = c_v;
          c_edge_weights[edge] = weight;
          ++edge;
        }

        map.clear();
      };

      const NodeID degree = c_nodes[c_u + 1] - c_nodes[c_u];
      local_collector.execute(degree, collect_edges);
    }
  });

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  return std::make_unique<CoarseGraphImpl<Mapping>>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(c_edges),
          std::move(c_node_weights),
          std::move(c_edge_weights)
      )),
      std::move(mapping)
  );
}

template <template <typename> typename Mapping, typename Graph, typename Clustering>
std::unique_ptr<CoarseGraph> contract_without_edgebuffer_remap(
    const Graph &graph,
    Clustering &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = preprocess<Mapping>(graph, clustering, m_ctx);

  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  // Overcomit memory for the edge and edge weight array as we only know the amount of edges of
  // the coarse graph afterwards.
  const EdgeID edge_count = graph.m();
  NodeID *c_edges;
  EdgeWeight *c_edge_weights;
  if constexpr (kHeapProfiling) {
    // As we overcommit memory do not track the amount of bytes used directly. Instead record it
    // manually afterwards.
    c_edges = (NodeID *)heap_profiler::std_malloc(edge_count * sizeof(NodeID));
    c_edge_weights = (EdgeWeight *)heap_profiler::std_malloc(edge_count * sizeof(EdgeWeight));
  } else {
    c_edges = (NodeID *)std::malloc(edge_count * sizeof(NodeID));
    c_edge_weights = (EdgeWeight *)std::malloc(edge_count * sizeof(EdgeWeight));
  }

  START_HEAP_PROFILER("Construct coarse graph");
  START_TIMER("Construct coarse graph");

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};
  CompactStaticArray<NodeID> remapping(math::byte_width(c_n), c_n);

  const auto write_neighbourhood = [&](const NodeID c_u,
                                       const NodeID new_c_u,
                                       EdgeID edge,
                                       const NodeWeight c_u_weight,
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

  __uint128_t next_coarse_node_info = 0;
  const auto &atomic_fetch_next_coarse_node_info = [&](std::uint64_t nodes, std::uint64_t degree) {
    std::uint64_t old_c_v;
    std::uint64_t old_edge;

    bool success;
    do {
      __uint128_t expected = next_coarse_node_info;
      old_c_v = (expected >> 64) & 0xFFFFFFFFFFFFFFFF;
      old_edge = expected & 0xFFFFFFFFFFFFFFFF;

      __uint128_t desired = (static_cast<__uint128_t>(old_c_v + nodes) << 64) |
                            static_cast<__uint128_t>(old_edge + degree);
      success = __sync_bool_compare_and_swap(&next_coarse_node_info, expected, desired);
    } while (!success);

    return std::make_pair(old_c_v, old_edge);
  };

  static constexpr NodeID kBufferSize = 30000;
  tbb::enumerable_thread_specific<std::tuple<
      NodeID,
      EdgeID,
      std::array<NodeID, kBufferSize>,
      std::array<EdgeID, kBufferSize>,
      std::array<NodeWeight, kBufferSize>,
      std::array<NodeID, kBufferSize>,
      std::array<EdgeWeight, kBufferSize>>>
      edge_buffer_ets;
  const auto flush_buffer = [&](NodeID num_buffered_nodes,
                                EdgeID num_buffered_edges,
                                auto &remapping_buffer,
                                auto &node_buffer,
                                auto &node_weight_buffer,
                                auto &edge_buffer,
                                auto &edge_weight_buffer,
                                NodeID new_c_u,
                                EdgeID edge) {
    std::memcpy(
        c_node_weights.data() + new_c_u,
        node_weight_buffer.data(),
        num_buffered_nodes * sizeof(NodeWeight)
    );
    std::memcpy(c_edges + edge, edge_buffer.data(), num_buffered_edges * sizeof(NodeID));
    std::memcpy(
        c_edge_weights + edge, edge_weight_buffer.data(), num_buffered_edges * sizeof(EdgeWeight)
    );

    for (NodeID i = 0; i < num_buffered_nodes; ++i) {
      remapping.write(remapping_buffer[i], new_c_u + i);

      c_nodes[new_c_u + i] = edge;
      edge += node_buffer[i];
    }
  };

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();
    auto
        &[num_buffered_nodes,
          num_buffered_edges,
          remapping_buffer,
          node_buffer,
          node_weight_buffer,
          edge_buffer,
          edge_weight_buffer] = edge_buffer_ets.local();

    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      const NodeID first = buckets_index[c_u];
      const NodeID last = buckets_index[c_u + 1];

      // Build coarse graph
      const auto collect_edges = [&](auto &map) {
        NodeWeight c_u_weight = 0;
        for (NodeID i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          KASSERT(mapping[u] == c_u);

          c_u_weight += graph.node_weight(u);

          graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += graph.edge_weight(e);
            }
          });
        }

        const std::size_t degree = map.size();
        if (degree >= kBufferSize) {
          auto [new_c_u, edge] = atomic_fetch_next_coarse_node_info(1, degree);
          write_neighbourhood(c_u, new_c_u, edge, c_u_weight, map);
        } else if (num_buffered_nodes >= kBufferSize - 1 ||
                   num_buffered_edges + degree >= kBufferSize) {
          const auto [new_c_u, edge] = atomic_fetch_next_coarse_node_info(
              num_buffered_nodes + 1, num_buffered_edges + degree
          );

          flush_buffer(
              num_buffered_nodes,
              num_buffered_edges,
              remapping_buffer,
              node_buffer,
              node_weight_buffer,
              edge_buffer,
              edge_weight_buffer,
              new_c_u,
              edge
          );

          write_neighbourhood(
              c_u, new_c_u + num_buffered_nodes, edge + num_buffered_edges, c_u_weight, map
          );

          num_buffered_nodes = 0;
          num_buffered_edges = 0;
        } else {
          remapping_buffer[num_buffered_nodes] = c_u;
          node_buffer[num_buffered_nodes] = degree;
          node_weight_buffer[num_buffered_nodes] = c_u_weight;
          num_buffered_nodes += 1;

          for (const auto [c_v, weight] : map.entries()) {
            edge_buffer[num_buffered_edges] = c_v;
            edge_weight_buffer[num_buffered_edges] = weight;
            num_buffered_edges += 1;
          }
        }

        map.clear();
      };

      // To select the right map, we need a upper bound on the coarse node degree. If we
      // previously split the coarse nodes into chunks, we have already computed them and stored
      // them in the c_nodes array.
      NodeID upper_bound_degree = 0;
      for (NodeID i = first; i < last; ++i) {
        const NodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }

      local_collector.execute(upper_bound_degree, collect_edges);
    }
  });

  tbb::parallel_for(edge_buffer_ets.range(), [&](auto &r) {
    for (auto
             &[num_buffered_nodes,
               num_buffered_edges,
               remapping_buffer,
               node_buffer,
               node_weight_buffer,
               edge_buffer,
               edge_weight_buffer] : r) {
      if (num_buffered_nodes > 0) {
        auto [new_c_u, edge] =
            atomic_fetch_next_coarse_node_info(num_buffered_nodes, num_buffered_edges);

        flush_buffer(
            num_buffered_nodes,
            num_buffered_edges,
            remapping_buffer,
            node_buffer,
            node_weight_buffer,
            edge_buffer,
            edge_weight_buffer,
            new_c_u,
            edge
        );
      }
    }
  });

  const EdgeID c_m = next_coarse_node_info & 0xFFFFFFFFFFFFFFFF;
  c_nodes[c_n] = c_m;

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
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Coarse graph edges allocation");
  RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, c_edges);
  RECORD("c_edge_weights") StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, c_edge_weights);
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(c_edges, c_m * sizeof(NodeID));
    heap_profiler::HeapProfiler::global().record_alloc(c_edge_weights, c_m * sizeof(EdgeWeight));
  }
  STOP_HEAP_PROFILER();

  return std::make_unique<CoarseGraphImpl<Mapping>>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(finalized_c_edges),
          std::move(c_node_weights),
          std::move(finalized_c_edge_weights)
      )),
      std::move(mapping)
  );
}

template <typename Graph, typename Clustering>
std::unique_ptr<CoarseGraph> contract_generic_graph(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    Clustering &clustering,
    MemoryContext &m_ctx
) {
  if (con_ctx.mode == ContractionMode::NO_EDGE_BUFFER_NAIVE) {
    return contract_without_edgebuffer_naive<CompactStaticArray>(graph, clustering, con_ctx, m_ctx);
  } else if (con_ctx.mode == ContractionMode::NO_EDGE_BUFFER_REMAP) {
    return contract_without_edgebuffer_remap<CompactStaticArray>(graph, clustering, con_ctx, m_ctx);
  } else {
    return contract_with_edgebuffer<CompactStaticArray>(graph, clustering, con_ctx, m_ctx);
  }
}
} // namespace

std::unique_ptr<CoarseGraph> contract(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    scalable_vector<parallel::Atomic<NodeID>> &clustering,
    MemoryContext &m_ctx
) {
  return graph.reified([&](const auto &concrete_graph) {
    return contract_generic_graph(concrete_graph, con_ctx, clustering, m_ctx);
  });
}
} // namespace kaminpar::shm::graph
