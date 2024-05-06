/*******************************************************************************
 * Graph contraction for local clusterings.
 *
 * @file:   local_cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#include "kaminpar-dist/coarsening/contraction/local_cluster_contraction.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(false);
}

namespace {
struct Edge {
  NodeID target;
  EdgeWeight weight;
};

class LocalCoarseGraphImpl : public CoarseGraph {
public:
  LocalCoarseGraphImpl(
      const DistributedGraph &f_graph, DistributedGraph c_graph, StaticArray<NodeID> mapping
  )
      : _f_graph(f_graph),
        _c_graph(std::move(c_graph)),
        _mapping(std::move(mapping)) {}

  const DistributedGraph &get() const final {
    return _c_graph;
  }

  DistributedGraph &get() final {
    return _c_graph;
  }

  void project(const StaticArray<BlockID> &c_partition, StaticArray<BlockID> &f_partition) final {
    TIMED_SCOPE("Project partition") {
      _f_graph.pfor_all_nodes([&](const NodeID u) { f_partition[u] = c_partition[_mapping[u]]; });
    };
    TIMER_BARRIER(_f_graph.communicator());
  }

private:
  const DistributedGraph &_f_graph;
  DistributedGraph _c_graph;
  StaticArray<NodeID> _mapping;
};
} // namespace

std::unique_ptr<CoarseGraph>
contract_local_clustering(const DistributedGraph &graph, const StaticArray<NodeID> &clustering) {
  KASSERT(
      clustering.size() >= graph.n(),
      "clustering array is too small for the given graph",
      assert::always
  );

  MPI_Comm comm = graph.communicator();
  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  //
  // Compute cluster buckets
  //

  StaticArray<NodeID> leader_mapping(graph.n());
  graph.pfor_nodes([&](const NodeID u) {
    __atomic_store_n(&leader_mapping[clustering[u]], 1, __ATOMIC_RELAXED);
  });
  parallel::prefix_sum(
      leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
  );

  const NodeID c_n = leader_mapping.back();
  const GlobalNodeID last_node = mpi::scan(static_cast<GlobalNodeID>(c_n), MPI_SUM, comm);
  const GlobalNodeID first_node = last_node - c_n;

  StaticArray<GlobalNodeID> c_node_distribution(size + 1);
  c_node_distribution[rank + 1] = last_node;
  mpi::allgather(&c_node_distribution[rank + 1], 1, c_node_distribution.data() + 1, 1, comm);

  StaticArray<NodeID> mapping(graph.total_n());
  graph.pfor_nodes([&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]] - 1; });

  StaticArray<NodeID> buckets_index(c_n + 1);
  graph.pfor_nodes([&](const NodeID u) {
    __atomic_fetch_add(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED);
  });
  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());

  StaticArray<NodeID> buckets(graph.n());
  graph.pfor_nodes([&](const NodeID u) {
    buckets[__atomic_sub_fetch(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED)] = u;
  });

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  StaticArray<EdgeID> c_nodes(c_n + 1);
  StaticArray<NodeWeight> c_node_weights(c_n);

  // Build coarse node weights
  tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
    const auto first = buckets_index[c_u];
    const auto last = buckets_index[c_u + 1];

    for (std::size_t i = first; i < last; ++i) {
      const NodeID u = buckets[i];
      c_node_weights[c_u] += graph.node_weight(u);
    }
  });

  //
  // Sparse all-to-all for building node mapping for ghost nodes
  //

  struct CoarseGhostNode {
    GlobalNodeID old_global_node;
    GlobalNodeID new_global_node;
    NodeWeight coarse_weight;
  };

  graph::GhostNodeMapper ghost_mapper(rank, c_node_distribution);

  mpi::graph::sparse_alltoall_interface_to_pe<CoarseGhostNode>(
      graph,
      [&](const NodeID u) -> CoarseGhostNode {
        KASSERT(u < mapping.size());
        KASSERT(mapping[u] < c_node_weights.size());
        return {
            .old_global_node = graph.local_to_global_node(u),
            .new_global_node = first_node + mapping[u],
            .coarse_weight = c_node_weights[mapping[u]],
        };
      },
      [&](const auto recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          const auto &[old_global_u, new_global_u, new_weight] = recv_buffer[i];
          const NodeID old_local_u = graph.global_to_local_node(old_global_u);
          mapping[old_local_u] = ghost_mapper.new_ghost_node(new_global_u);
        });
      }
  );

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
  using Map = RatingMap<EdgeWeight, NodeID, FastResetArray<EdgeWeight, NodeID>>;
  tbb::enumerable_thread_specific<Map> collector_ets{[&] {
    return Map(ghost_mapper.next_ghost_node());
  }};
  NavigableLinkedList<NodeID, Edge, ScalableVector> edge_buffer_ets;

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector_ets.local();
    auto &local_edge_buffer = edge_buffer_ets.local();

    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      local_edge_buffer.mark(c_u);

      const std::size_t first = buckets_index[c_u];
      const std::size_t last = buckets_index[c_u + 1];

      // build coarse graph
      auto collect_edges = [&](auto &map) {
        for (std::size_t i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          KASSERT(mapping[u] == c_u);

          // collect coarse edges
          for (const auto [e, v] : graph.neighbors(u)) {
            const NodeID c_v = mapping[v];
            if (c_u != c_v) {
              map[c_v] += graph.edge_weight(e);
            }
          }
        }

        c_nodes[c_u + 1] = map.size(); // node degree (used to build c_nodes)

        // since we don't know the value of c_nodes[c_u] yet (so far, it only
        // holds the nodes degree), we can't place the edges of c_u in the
        // c_edges and c_edge_weights arrays; hence, we store them in auxiliary
        // arrays and note their position in the auxiliary arrays
        for (const auto [c_v, weight] : map.entries()) {
          local_edge_buffer.push_back({c_v, weight});
        }
        map.clear();
      };

      // to select the right map, we compute an upper bound on the coarse node
      // degree by summing the degree of all fine nodes
      EdgeID upper_bound_degree = 0;
      for (std::size_t i = first; i < last; ++i) {
        const NodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }
      local_collector.execute(upper_bound_degree, collect_edges);
    }
  });

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //
  StaticArray<NavigationMarker<NodeID, Edge, ScalableVector>> all_buffered_nodes;
  all_buffered_nodes = ts_navigable_list::combine<NodeID, Edge, ScalableVector>(
      edge_buffer_ets, std::move(all_buffered_nodes)
  );

  StaticArray<NodeID> c_edges(c_m);
  StaticArray<EdgeWeight> c_edge_weights(c_m);

  // build coarse graph
  tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID i) {
    const auto &marker = all_buffered_nodes[i];
    const auto *list = marker.local_list;
    const NodeID c_u = marker.key;

    const EdgeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
    const EdgeID first_target_index = c_nodes[c_u];
    const EdgeID first_source_index = marker.position;

    for (std::size_t j = 0; j < c_u_degree; ++j) {
      const auto to = first_target_index + j;
      const auto [c_v, weight] = list->get(first_source_index + j);
      c_edges[to] = c_v;
      c_edge_weights[to] = weight;
    }
  });

  // compute edge distribution
  const GlobalEdgeID last_edge = mpi::scan(static_cast<GlobalEdgeID>(c_m), MPI_SUM, comm);
  StaticArray<GlobalEdgeID> c_edge_distribution(size + 1);
  c_edge_distribution[rank + 1] = last_edge;
  mpi::allgather(&c_edge_distribution[rank + 1], 1, c_edge_distribution.data() + 1, 1, comm);

  auto [c_global_to_ghost, c_ghost_to_global, c_ghost_owner] = ghost_mapper.finalize();

  DistributedGraph c_graph{
      std::move(c_node_distribution),
      std::move(c_edge_distribution),
      std::move(c_nodes),
      std::move(c_edges),
      std::move(c_node_weights),
      std::move(c_edge_weights),
      std::move(c_ghost_owner),
      std::move(c_ghost_to_global),
      std::move(c_global_to_ghost),
      false,
      graph.communicator()
  };

  return std::make_unique<LocalCoarseGraphImpl>(graph, std::move(c_graph), std::move(mapping));
}
} // namespace kaminpar::dist
