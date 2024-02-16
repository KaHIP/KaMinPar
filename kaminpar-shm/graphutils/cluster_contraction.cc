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

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
using namespace contraction;

namespace {
template <typename Clustering>
Result
contract_generic_clustering(const Graph &graph, const Clustering &clustering, MemoryContext m_ctx) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  scalable_vector<NodeID> mapping(graph.n());
  if (leader_mapping.size() < graph.n()) {
    leader_mapping.resize(graph.n());
  }
  if (buckets.size() < graph.n()) {
    buckets.resize(graph.n());
  }
  STOP_TIMER();

  START_TIMER("Preprocessing");

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the
  // coarse graph I.e., node_mapping[node u] = coarse node c_u
  //
  // Note that clustering satisfies this invariant (I): if clustering[x] = y for
  // some node x, then clustering[y] = y
  //

  // Set node_mapping[x] = 1 iff. there is a cluster with leader x
  graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const NodeID u) {
    leader_mapping[clustering[u]].store(1, std::memory_order_relaxed);
  });

  // Compute prefix sum to get coarse node IDs (starting at 1!)
  parallel::prefix_sum(
      leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
  );
  const NodeID c_n = leader_mapping[graph.n() - 1]; // number of nodes in the coarse graph

  // Assign coarse node ID to all nodes; this works due to (I)
  graph.pfor_nodes([&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]]; });
  graph.pfor_nodes([&](const NodeID u) { --mapping[u]; });

  STOP_TIMER();

  TIMED_SCOPE("Allocation") {
    buckets_index.clear();
    buckets_index.resize(c_n + 1);
  };

  START_TIMER("Preprocessing");

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into
  // the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the
  // bucket in the global buckets array using a prefix sum, roughly 2/5-th of
  // time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
  graph.pfor_nodes([&](const NodeID u) {
    buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed);
  });

  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  KASSERT(buckets_index.back() <= graph.n());

  // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  STOP_TIMER(); // Preprocessing

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
          for (const auto [e, v] : graph.neighbors(u)) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += graph.edge_weight(e);
            }
          }
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
  StaticArray<NodeID> c_edges{c_m};
  StaticArray<EdgeWeight> c_edge_weights{c_m};
  STOP_TIMER();

  // build coarse graph
  START_TIMER("Construct coarse graph");
  tbb::parallel_for(static_cast<NodeID>(0), c_n, [&](const NodeID i) {
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

  return {
      Graph{
          std::move(c_nodes),
          std::move(c_edges),
          std::move(c_node_weights),
          std::move(c_edge_weights)},
      std::move(mapping),
      std::move(m_ctx)};
}
} // namespace

Result
contract(const Graph &graph, const scalable_vector<NodeID> &clustering, MemoryContext m_ctx) {
  return contract_generic_clustering(graph, clustering, std::move(m_ctx));
}

Result contract(
    const Graph &graph,
    const scalable_vector<parallel::Atomic<NodeID>> &clustering,
    MemoryContext m_ctx
) {
  return contract_generic_clustering(graph, clustering, std::move(m_ctx));
}
} // namespace kaminpar::shm::graph
