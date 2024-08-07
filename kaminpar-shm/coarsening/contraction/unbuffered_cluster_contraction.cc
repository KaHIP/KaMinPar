/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.cc
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h"

#include <memory>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {
namespace {
class NeighborhoodsBuffer {
  static constexpr NodeID kSize = 30000; // Chosen such that its about 1 MB in size

public:
  [[nodiscard]] static bool exceeds_capacity(const NodeID degree) {
    return degree >= kSize;
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
        _remapping(remapping) {}

  [[nodiscard]] NodeID num_buffered_nodes() const {
    return _num_buffered_nodes;
  }

  [[nodiscard]] NodeID num_buffered_edges() const {
    return _num_buffered_edges;
  }

  [[nodiscard]] bool overfills(const NodeID degree) const {
    return _num_buffered_nodes + 1 >= kSize || _num_buffered_edges + degree >= kSize;
  }

  template <typename Lambda>
  void add(const NodeID c_u, const NodeID degree, const NodeWeight weight, Lambda &&it) {
    _remapping_buffer[_num_buffered_nodes] = c_u;
    _node_buffer[_num_buffered_nodes] = degree;
    _node_weight_buffer[_num_buffered_nodes] = weight;
    _num_buffered_nodes += 1;

    it([this](const auto c_v, const auto weight) {
      _edge_buffer[_num_buffered_edges] = c_v;
      _edge_weight_buffer[_num_buffered_edges] = weight;
      _num_buffered_edges += 1;
    });
  }

  void flush(const NodeID c_u, EdgeID e) {
    std::memcpy(
        _node_weights + c_u, _node_weight_buffer.data(), _num_buffered_nodes * sizeof(NodeWeight)
    );
    std::memcpy(_edges + e, _edge_buffer.data(), _num_buffered_edges * sizeof(NodeID));
    std::memcpy(
        _edge_weights + e, _edge_weight_buffer.data(), _num_buffered_edges * sizeof(EdgeWeight)
    );

    for (NodeID i = 0; i < _num_buffered_nodes; ++i) {
      _remapping.write(_remapping_buffer[i], c_u + i);

      _nodes[c_u + i] = e;
      e += _node_buffer[i];
    }

    _num_buffered_nodes = 0;
    _num_buffered_edges = 0;
  }

private:
  EdgeID *_nodes;
  NodeID *_edges;
  NodeWeight *_node_weights;
  EdgeWeight *_edge_weights;
  CompactStaticArray<NodeID> &_remapping;

  NodeID _num_buffered_nodes{0};
  EdgeID _num_buffered_edges{0};
  std::array<NodeID, kSize> _remapping_buffer;
  std::array<EdgeID, kSize> _node_buffer;
  std::array<NodeWeight, kSize> _node_weight_buffer;
  std::array<NodeID, kSize> _edge_buffer;
  std::array<EdgeWeight, kSize> _edge_weight_buffer;
};

template <typename Graph>
std::unique_ptr<CoarseGraph> contract_clustering_unbuffered(
    const Graph &graph,
    const NodeID c_n,
    StaticArray<NodeID> mapping,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1, static_array::noinit);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n, static_array::noinit);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  // Overcomit memory for the edge and edge weight array as we only know the amount of edges of
  // the coarse graph afterwards.
  const EdgeID num_fine_edges = graph.m();
  auto c_edges = heap_profiler::overcommit_memory<NodeID>(num_fine_edges);
  auto c_edge_weights = heap_profiler::overcommit_memory<EdgeWeight>(num_fine_edges);

  START_HEAP_PROFILER("Construct coarse graph");
  START_TIMER("Construct coarse graph");

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
      c_edges.get()[edge] = c_v;
      c_edge_weights.get()[edge] = weight;
      edge += 1;
    }
  };

  __uint128_t next_coarse_node_info = 0;
  const auto atomic_fetch_next_coarse_node_info = [&](const std::uint64_t nodes,
                                                      const std::uint64_t degree) {
    std::uint64_t c_v;
    std::uint64_t edge;

    bool success;
    do {
      const __uint128_t expected = next_coarse_node_info;
      c_v = (expected >> 64) & 0xFFFFFFFFFFFFFFFF;
      edge = expected & 0xFFFFFFFFFFFFFFFF;

      const __uint128_t desired =
          (static_cast<__uint128_t>(c_v + nodes) << 64) | static_cast<__uint128_t>(edge + degree);
      success = __sync_bool_compare_and_swap(&next_coarse_node_info, expected, desired);
    } while (!success);

    return std::make_pair(c_v, edge);
  };

  const auto aggregate_edges = [&](const NodeID c_u,
                                   const NodeID first,
                                   const NodeID last,
                                   auto &edge_collector,
                                   auto &neighborhood_buffer) {
    NodeWeight c_u_weight = 0;
    for (NodeID i = first; i < last; ++i) {
      const NodeID u = buckets[i];
      KASSERT(mapping[u] == c_u);

      c_u_weight += graph.node_weight(u);

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const NodeID c_v = mapping[v];
        if (c_u != c_v) {
          edge_collector[c_v] += w;
        }
      });
    }

    const std::size_t degree = edge_collector.size();
    if (NeighborhoodsBuffer::exceeds_capacity(degree)) {
      auto [new_c_u, edge] = atomic_fetch_next_coarse_node_info(1, degree);
      write_neighbourhood(c_u, c_u_weight, new_c_u, edge, edge_collector);
    } else if (neighborhood_buffer.overfills(degree)) {
      const NodeID num_buffered_nodes = neighborhood_buffer.num_buffered_nodes();
      const EdgeID num_buffered_edges = neighborhood_buffer.num_buffered_edges();
      const auto [new_c_u, edge] =
          atomic_fetch_next_coarse_node_info(num_buffered_nodes + 1, num_buffered_edges + degree);
      neighborhood_buffer.flush(new_c_u, edge);
      write_neighbourhood(
          c_u, c_u_weight, new_c_u + num_buffered_nodes, edge + num_buffered_edges, edge_collector
      );
    } else {
      neighborhood_buffer.add(c_u, degree, c_u_weight, [&](auto &&l) {
        for (const auto [c_v, weight] : edge_collector.entries()) {
          l(c_v, weight);
        }
      });
    }

    edge_collector.clear();
  };

  tbb::enumerable_thread_specific<NeighborhoodsBuffer> neighborhoods_buffer_ets{[&] {
    return NeighborhoodsBuffer(
        c_nodes.data(), c_edges.get(), c_node_weights.data(), c_edge_weights.get(), remapping
    );
  }};

  if (con_ctx.use_growing_hash_tables) {
    using EdgeCollector = DynamicRememberingFlatMap<NodeID, EdgeWeight, NodeID>;
    tbb::enumerable_thread_specific<EdgeCollector> edge_collector_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_collector = edge_collector_ets.local();
      auto &local_buffer = neighborhoods_buffer_ets.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];
        aggregate_edges(c_u, first, last, local_collector, local_buffer);
      }
    });
  } else {
    using EdgeCollector = RatingMap<EdgeWeight, NodeID>;
    tbb::enumerable_thread_specific<EdgeCollector> edge_collector_ets{[&] {
      return EdgeCollector(c_n);
    }};

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_collector = edge_collector_ets.local();
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

        local_collector.execute(upper_bound_degree, [&](auto &edge_collector) {
          aggregate_edges(c_u, first, last, edge_collector, local_buffer);
        });
      }
    });
  }

  tbb::parallel_for(neighborhoods_buffer_ets.range(), [&](auto &r) {
    for (auto &buffer : r) {
      const NodeID num_buffered_nodes = buffer.num_buffered_nodes();
      if (num_buffered_nodes == 0) {
        continue;
      }

      const EdgeID num_buffered_edges = buffer.num_buffered_edges();
      auto [new_c_u, edge] =
          atomic_fetch_next_coarse_node_info(num_buffered_nodes, num_buffered_edges);
      buffer.flush(new_c_u, edge);
    }
  });

  const EdgeID c_m = next_coarse_node_info & 0xFFFFFFFFFFFFFFFF;
  c_nodes[c_n] = c_m;

  tbb::parallel_for(tbb::blocked_range<EdgeID>(0, c_m), [&](const auto &r) {
    for (EdgeID e = r.begin(); e != r.end(); ++e) {
      c_edges.get()[e] = remapping[c_edges.get()[e]];
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
  RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, std::move(c_edges));
  RECORD("c_edge_weights")
  StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, std::move(c_edge_weights));
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(c_edges.get(), c_m * sizeof(NodeID));
    heap_profiler::HeapProfiler::global().record_alloc(
        c_edge_weights.get(), c_m * sizeof(EdgeWeight)
    );
  }
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
  return graph.reified([&](auto &graph) {
    return contract_clustering_unbuffered(graph, c_n, std::move(mapping), con_ctx, m_ctx);
  });
}
} // namespace kaminpar::shm::contraction
