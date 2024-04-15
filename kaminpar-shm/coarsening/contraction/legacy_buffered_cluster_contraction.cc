/*******************************************************************************
 * Contraction implementation that uses an edge buffer to store edges before
 * building the final graph.
 *
 * @file:   legacy_buffered_cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/legacy_buffered_cluster_contraction.h"

#include <memory>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {
namespace {
template <template <typename> typename Mapping, typename Graph>
std::unique_ptr<CoarseGraph> contract_with_edgebuffer_legacy(
    const Graph &graph,
    const NodeID c_n,
    Mapping<NodeID> mapping,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  START_TIMER("Allocation");
  StaticArray<EdgeID> c_nodes{c_n + 1};
  StaticArray<NodeWeight> c_node_weights{c_n};
  STOP_TIMER();

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};

  //
  // We build the coarse graph in multiple steps:
  // (1) During the first step, we compute
  //     - the node weight of each coarse node
  //     - the degree of each coarse node
  //     We can't build c_edges and c_edge_weights yet, because positioning
  //     edges in those arrays depends on c_nodes, which we only have after
  //     computing a prefix sum over all coarse node degrees Hence, we store
  //     edges and edge weights in unsorted auxiliary arrays during the first
  //     pass
  // (2) We finalize c_nodes arrays by computing a prefix sum over all coarse
  // node degrees (3) We copy coarse edges and coarse edge weights from the
  // auxiliary arrays to c_edges and c_edge_weights
  //
  NavigableLinkedList<NodeID, Edge, scalable_vector> edge_buffer_ets;

  START_TIMER("Construct coarse edges");
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();
    auto &local_edge_buffer = edge_buffer_ets.local();

    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      local_edge_buffer.mark(c_u);

      const std::size_t first = buckets_index[c_u];
      const std::size_t last = buckets_index[c_u + 1];

      // build coarse graph
      auto collect_edges = [&](auto &map) {
        NodeWeight c_u_weight = 0;
        for (std::size_t i = first; i < last; ++i) {
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

        // since we don't know the value of c_nodes[c_u] yet (so far, it only
        // holds the nodes degree), we can't place the edges of c_u in the
        // c_edges and c_edge_weights arrays; hence, we store them in auxiliary
        // arrays and note their position in the auxiliary arrays
        for (const auto [c_v, weight] : map.entries()) {
          local_edge_buffer.push_back({c_v, weight});
        }
        map.clear();
      };

      // to select the right map, we compute a upper bound on the coarse node
      // degree by summing the degree of all fine nodes
      NodeID upper_bound_degree = 0;
      for (std::size_t i = first; i < last; ++i) {
        const NodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }
      local_collector.execute(upper_bound_degree, collect_edges);
    }
  });

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());
  STOP_TIMER(); // Graph construction

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //

  all_buffered_nodes = ts_navigable_list::combine<NodeID, Edge, scalable_vector>(
      edge_buffer_ets, std::move(all_buffered_nodes)
  );

  START_TIMER("Allocation");
  StaticArray<NodeID> c_edges(c_m);
  StaticArray<EdgeWeight> c_edge_weights(c_m);
  STOP_TIMER();

  // build coarse graph
  START_TIMER("Construct coarse graph");
  tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID i) {
    const auto &marker = all_buffered_nodes[i];
    const auto *list = marker.local_list;
    const NodeID c_u = marker.key;

    const NodeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
    const EdgeID first_target_index = c_nodes[c_u];
    const EdgeID first_source_index = marker.position;

    for (std::size_t j = 0; j < c_u_degree; ++j) {
      const auto to = first_target_index + j;
      const auto [c_v, weight] = list->get(first_source_index + j);
      c_edges[to] = c_v;
      c_edge_weights[to] = weight;
    }
  });
  STOP_TIMER();

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
} // namespace

std::unique_ptr<CoarseGraph> contract_with_edgebuffer_legacy(
    const Graph &graph,
    StaticArray<NodeID> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  if (con_ctx.use_compact_mapping) {
    auto [c_n, mapping] = preprocess<CompactStaticArray>(graph, clustering, m_ctx);
    return graph.reified([&](auto &graph) {
      return contract_with_edgebuffer_legacy(graph, c_n, std::move(mapping), con_ctx, m_ctx);
    });
  } else {
    auto [c_n, mapping] = preprocess<StaticArray>(graph, clustering, m_ctx);
    return graph.reified([&](auto &graph) {
      return contract_with_edgebuffer_legacy(graph, c_n, std::move(mapping), con_ctx, m_ctx);
    });
  }
}
} // namespace kaminpar::shm::contraction

