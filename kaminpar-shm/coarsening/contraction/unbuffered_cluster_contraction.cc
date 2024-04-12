/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.cc
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h"

#include <memory>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {
namespace {
template <template <typename> typename Mapping, typename Graph>
std::unique_ptr<CoarseGraph> contract_without_edgebuffer_remap(
    const Graph &graph,
    const NodeID c_n,
    Mapping<NodeID> mapping,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
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
} // namespace

std::unique_ptr<CoarseGraph> contract_without_edgebuffer_remap(
    const Graph &graph,
    scalable_vector<parallel::Atomic<NodeID>> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = preprocess<CompactStaticArray>(graph, clustering, m_ctx);
  return graph.reified([&](auto &graph) {
    return contract_without_edgebuffer_remap(graph, c_n, std::move(mapping), con_ctx, m_ctx);
  });
}
} // namespace kaminpar::shm::contraction
