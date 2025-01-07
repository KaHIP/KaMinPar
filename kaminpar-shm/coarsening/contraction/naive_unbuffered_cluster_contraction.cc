/*******************************************************************************
 * @file:   naive_unbuffered_cluster_contraction.cc
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#include <memory>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {

namespace {

template <typename Graph>
std::unique_ptr<CoarseGraph> contract_clustering_unbuffered_naive(
    const Graph &graph,
    const NodeID c_n,
    StaticArray<NodeID> mapping,
    [[maybe_unused]] const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;

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
          graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += w;
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
  RECORD("c_edges") StaticArray<NodeID> c_edges(c_m, static_array::noinit);
  RECORD("c_edge_weights")
  StaticArray<EdgeWeight> c_edge_weights(c_m, static_array::noinit);
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
          graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += w;
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

  return std::make_unique<CoarseGraphImpl>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(c_edges),
          std::move(c_node_weights),
          std::move(c_edge_weights)
      )),
      std::move(mapping)
  );
}

} // namespace

std::unique_ptr<CoarseGraph> contract_clustering_unbuffered_naive(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = compute_mapping(graph, std::move(clustering), m_ctx);
  fill_cluster_buckets(c_n, graph, mapping, m_ctx.buckets_index, m_ctx.buckets);
  return graph.reified([&](auto &graph) {
    return contract_clustering_unbuffered_naive(graph, c_n, std::move(mapping), con_ctx, m_ctx);
  });
}

} // namespace kaminpar::shm::contraction
