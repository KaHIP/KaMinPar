/*******************************************************************************
 * Graph contraction for arbitrary clusterings.
 *
 * In this file, we use the following naming sheme for node and cluster IDs:
 * - {g,l}[c]{node,cluster}
 *    ^ global or local ID
 *         ^ ID in [c]oarse graph or in fine graph
 *            ^ node or cluster ID
 *
 * @file:   cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 * @brief:  Graph contraction for arbitrary clusterings.
 ******************************************************************************/
#include "kaminpar-dist/coarsening/contraction/cluster_contraction.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>

#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/aligned_element.h"
#include "kaminpar-common/parallel/vector_ets.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

ContractionResult contract_clustering(
    const DistributedGraph &graph, GlobalClustering &clustering, const CoarseningContext &c_ctx
) {
  return contract_clustering(
      graph,
      clustering,
      c_ctx.max_cnode_imbalance,
      c_ctx.migrate_cnode_prefix,
      c_ctx.force_perfect_cnode_balance
  );
}

namespace {
struct AssignmentShifts {
  NoinitVector<GlobalNodeID> overload;
  NoinitVector<GlobalNodeID> underload;
};

struct GlobalEdge {
  GlobalNodeID u;
  GlobalNodeID v;
  EdgeWeight weight;
};

struct GlobalNode {
  GlobalNodeID u;
  NodeWeight weight;
};

template <typename T> struct MigrationResult {
  NoinitVector<T> elements;

  // Can be re-used for mapping exchange ...
  std::vector<int> sendcounts;
  std::vector<int> sdispls;
  std::vector<int> recvcounts;
  std::vector<int> rdispls;
};

struct NodeMapping {
  GlobalNodeID gcluster;
  GlobalNodeID gcnode;
};

struct MigratedNodesMapping {
  NoinitVector<NodeMapping> my_nonlocal_to_gcnode;
  NoinitVector<NodeID> their_req_to_lcnode;
};

NoinitVector<GlobalNode>
find_nonlocal_nodes(const DistributedGraph &graph, const GlobalClustering &lnode_to_gcluster) {
  SCOPED_TIMER("Collect nonlocal nodes");

  NoinitVector<NodeID> node_position_buffer(graph.n() + 1);
  node_position_buffer.front() = 0;
  graph.pfor_nodes([&](const NodeID lnode) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
    node_position_buffer[lnode + 1] = !graph.is_owned_global_node(gcluster);
  });
  parallel::prefix_sum(
      node_position_buffer.begin(), node_position_buffer.end(), node_position_buffer.begin()
  );

  NoinitVector<GlobalNode> nonlocal_nodes(node_position_buffer.back());
  graph.pfor_nodes([&](const NodeID lnode) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
    if (!graph.is_owned_global_node(gcluster)) {
      nonlocal_nodes[node_position_buffer[lnode]] = {
          .u = gcluster, .weight = graph.node_weight(lnode)};
    }
  });

  return nonlocal_nodes;
}

NoinitVector<GlobalEdge>
find_nonlocal_edges(const DistributedGraph &graph, const GlobalClustering &lnode_to_gcluster) {
  SCOPED_TIMER("Collect nonlocal edges");

  NoinitVector<NodeID> edge_position_buffer(graph.n() + 1);
  edge_position_buffer.front() = 0;

  graph.pfor_nodes([&](const NodeID lnode_u) {
    const GlobalNodeID gcluster_u = lnode_to_gcluster[lnode_u];

    NodeID nonlocal_neighbors_count = 0;
    if (!graph.is_owned_global_node(gcluster_u)) {
      for (const auto [e, lnode_v] : graph.neighbors(lnode_u)) {
        const GlobalNodeID gcluster_v = lnode_to_gcluster[lnode_v];
        if (gcluster_u != gcluster_v) {
          ++nonlocal_neighbors_count;
        }
      }
    }

    edge_position_buffer[lnode_u + 1] = nonlocal_neighbors_count;
  });

  parallel::prefix_sum(
      edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin()
  );

  NoinitVector<GlobalEdge> nonlocal_edges(edge_position_buffer.back());
  graph.pfor_nodes([&](const NodeID lnode_u) {
    const GlobalNodeID gcluster_u = lnode_to_gcluster[lnode_u];

    if (!graph.is_owned_global_node(gcluster_u)) {
      NodeID pos = edge_position_buffer[lnode_u];
      for (const auto [e, lnode_v] : graph.neighbors(lnode_u)) {
        const GlobalNodeID gcluster_v = lnode_to_gcluster[lnode_v];
        if (gcluster_u != gcluster_v) {
          nonlocal_edges[pos] = {
              .u = gcluster_u,
              .v = gcluster_v,
              .weight = graph.edge_weight(e),
          };
          ++pos;
        }
      }
    }
  });

  return nonlocal_edges;
}

void deduplicate_edge_list(NoinitVector<GlobalEdge> &edges) {
  SCOPED_TIMER("Deduplicate edge list");

  if (edges.empty()) {
    return;
  }

  // Primary sort by edge source = messages are sorted by destination PE
  // Secondary sort by edge target = duplicate edges are consecutive
  START_TIMER("Sort edges");
  tbb::parallel_sort(edges.begin(), edges.end(), [&](const auto &lhs, const auto &rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
  });
  STOP_TIMER();

  // Mark the first edge in every block of duplicate edges
  START_TIMER("Mark start of parallel edge blocks");
  NoinitVector<EdgeID> edge_position_buffer(edges.size());
  edge_position_buffer.front() = 0;
  tbb::parallel_for<std::size_t>(1, edges.size(), [&](const std::size_t i) {
    edge_position_buffer[i] = (edges[i].u != edges[i - 1].u || edges[i].v != edges[i - 1].v);
  });
  STOP_TIMER();

  // Prefix sum to get the location of the deduplicated edge
  parallel::prefix_sum(
      edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin()
  );

  // Deduplicate edges in a separate buffer
  START_TIMER("Deduplicate");
  NoinitVector<GlobalEdge> tmp_nonlocal_edges(edge_position_buffer.back() + 1);
  tbb::parallel_for<std::size_t>(0, edge_position_buffer.back() + 1, [&](const std::size_t i) {
    tmp_nonlocal_edges[i].weight = 0;
  });
  tbb::parallel_for<std::size_t>(0, edges.size(), [&](const std::size_t i) {
    const std::size_t pos = edge_position_buffer[i];
    __atomic_store_n(&(tmp_nonlocal_edges[pos].u), edges[i].u, __ATOMIC_RELAXED);
    __atomic_store_n(&(tmp_nonlocal_edges[pos].v), edges[i].v, __ATOMIC_RELAXED);
    __atomic_fetch_add(&(tmp_nonlocal_edges[pos].weight), edges[i].weight, __ATOMIC_RELAXED);
  });
  std::swap(tmp_nonlocal_edges, edges);
  STOP_TIMER();
}

void sort_node_list(NoinitVector<GlobalNode> &nodes) {
  SCOPED_TIMER("Sort nodes");

  tbb::parallel_sort(nodes.begin(), nodes.end(), [&](const GlobalNode &lhs, const GlobalNode &rhs) {
    return lhs.u < rhs.u;
  });
}

void update_ghost_node_weights(DistributedGraph &graph) {
  SCOPED_TIMER("Update ghost node weights");

  struct Message {
    NodeID local_node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID u) -> Message {
        return {u, graph.node_weight(u)};
      },
      [&](const auto buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &[local_node_on_other_pe, weight] = buffer[i];
          const NodeID local_node =
              graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
          graph.set_ghost_node_weight(local_node, weight);
        });
      }
  );
}

template <typename T> StaticArray<T> build_distribution(const T count, MPI_Comm comm) {
  SCOPED_TIMER("Build node distribution");

  const PEID size = mpi::get_comm_size(comm);
  StaticArray<T> distribution(size + 1);
  MPI_Allgather(
      &count,
      1,
      mpi::type::get<NodeID>(),
      distribution.data(),
      1,
      mpi::type::get<GlobalNodeID>(),
      comm
  );
  std::exclusive_scan(
      distribution.begin(), distribution.end(), distribution.begin(), static_cast<T>(0)
  );
  return distribution;
}

template <typename T> double compute_distribution_imbalance(const StaticArray<T> &distribution) {
  T max = 0;
  for (std::size_t i = 0; i + 1 < distribution.size(); ++i) {
    max = std::max(max, distribution[i + 1] - distribution[i]);
  }
  return 1.0 * max / (1.0 * distribution.back() / (distribution.size() - 1));
}

NoinitVector<NodeID> build_lcluster_to_lcnode_mapping(
    const DistributedGraph &graph,
    const GlobalClustering &lnode_to_gcluster,
    const NoinitVector<GlobalNode> &local_nodes
) {
  SCOPED_TIMER("Build lcluster_to_lcnode");

  NoinitVector<NodeID> lcluster_to_lcnode(graph.n());
  graph.pfor_nodes([&](const NodeID u) { lcluster_to_lcnode[u] = 0; });
  tbb::parallel_invoke(
      [&] {
        graph.pfor_nodes([&](const NodeID lnode) {
          const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
          if (graph.is_owned_global_node(gcluster)) {
            const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
            __atomic_store_n(&lcluster_to_lcnode[lcluster], 1, __ATOMIC_RELAXED);
          }
        });
      },
      [&] {
        tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
          const GlobalNodeID gcluster = local_nodes[i].u;
          KASSERT(graph.is_owned_global_node(gcluster));
          const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
          __atomic_store_n(&lcluster_to_lcnode[lcluster], 1, __ATOMIC_RELAXED);
        });
      }
  );
  parallel::prefix_sum(
      lcluster_to_lcnode.begin(), lcluster_to_lcnode.end(), lcluster_to_lcnode.begin()
  );
  tbb::parallel_for<std::size_t>(0, lcluster_to_lcnode.size(), [&](const std::size_t i) {
    lcluster_to_lcnode[i] -= 1;
  });

  return lcluster_to_lcnode;
}

void localize_global_edge_list(
    NoinitVector<GlobalEdge> &edges,
    const GlobalNodeID offset,
    const NoinitVector<NodeID> &lnode_to_lcnode
) {
  tbb::parallel_for<std::size_t>(0, edges.size(), [&](const std::size_t i) {
    const NodeID lcluster = static_cast<NodeID>(edges[i].u - offset);
    edges[i].u = lnode_to_lcnode[lcluster];
  });
}

std::pair<NoinitVector<NodeID>, NoinitVector<NodeID>> build_node_buckets(
    const DistributedGraph &graph,
    const NoinitVector<NodeID> &lcluster_to_lcnode,
    const GlobalNodeID c_n,
    const NoinitVector<GlobalEdge> &local_edges,
    const GlobalClustering &lnode_to_gcluster
) {
  SCOPED_TIMER("Bucket sort nodes by clusters");

  NoinitVector<NodeID> buckets_position_buffer(c_n + 1);
  tbb::parallel_for<NodeID>(0, c_n + 1, [&](const NodeID lcnode) {
    buckets_position_buffer[lcnode] = 0;
  });

  tbb::parallel_invoke(
      [&] {
        graph.pfor_nodes([&](const NodeID lnode) {
          const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
          if (graph.is_owned_global_node(gcluster)) {
            const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
            const NodeID lcnode = lcluster_to_lcnode[lcluster];
            KASSERT(lcnode < buckets_position_buffer.size());
            __atomic_fetch_add(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
          }
        });
      },
      [&] {
        tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
          if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
            __atomic_fetch_add(&buckets_position_buffer[local_edges[i].u], 1, __ATOMIC_RELAXED);
          }
        });
      }
  );

  parallel::prefix_sum(
      buckets_position_buffer.begin(),
      buckets_position_buffer.end(),
      buckets_position_buffer.begin()
  );

  NoinitVector<NodeID> buckets(
      buckets_position_buffer.empty() ? 0 : buckets_position_buffer.back()
  );
  tbb::parallel_invoke(
      [&] {
        graph.pfor_nodes([&](const NodeID lnode) {
          const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
          if (graph.is_owned_global_node(gcluster)) {
            const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
            const NodeID lcnode = lcluster_to_lcnode[lcluster];
            const std::size_t pos =
                __atomic_fetch_sub(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
            buckets[pos - 1] = lnode;
          }
        });
      },
      [&] {
        tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
          if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
            const NodeID lcnode = local_edges[i].u;
            const std::size_t pos =
                __atomic_fetch_sub(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
            buckets[pos - 1] = graph.n() + i;
          }
        });
      }
  );

  return {std::move(buckets_position_buffer), std::move(buckets)};
}

template <typename Element, typename NumElementsForPEContainer>
MigrationResult<Element> migrate_elements(
    const NumElementsForPEContainer &num_elements_for_pe,
    const NoinitVector<Element> &elements,
    MPI_Comm comm
) {
  const PEID size = mpi::get_comm_size(comm);

  NoinitVector<Element> recvbuf;
  std::vector<int> sendcounts(size);
  std::vector<int> sdispls(size);
  std::vector<int> recvcounts(size);
  std::vector<int> rdispls(size);

  std::copy(num_elements_for_pe.begin(), num_elements_for_pe.end(), sendcounts.begin());
  std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
  std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

  recvbuf.resize(rdispls.back() + recvcounts.back());
  MPI_Alltoallv(
      elements.data(),
      sendcounts.data(),
      sdispls.data(),
      mpi::type::get<Element>(),
      recvbuf.data(),
      recvcounts.data(),
      rdispls.data(),
      mpi::type::get<Element>(),
      comm
  );

  return {
      .elements = std::move(recvbuf),
      .sendcounts = std::move(sendcounts),
      .sdispls = std::move(sdispls),
      .recvcounts = std::move(recvcounts),
      .rdispls = std::move(rdispls)};
}

MigrationResult<GlobalNode>
migrate_nodes(const DistributedGraph &graph, const NoinitVector<GlobalNode> &nonlocal_nodes) {
  SCOPED_TIMER("Exchange nonlocal nodes");

  const PEID size = mpi::get_comm_size(graph.communicator());

  START_TIMER("Count messages");
  parallel::vector_ets<NodeID> num_nodes_for_pe_ets(size);
  tbb::parallel_for<std::size_t>(0, nonlocal_nodes.size(), [&](const std::size_t i) {
    auto &num_nodes_for_pe = num_nodes_for_pe_ets.local();
    const PEID pe = graph.find_owner_of_global_node(nonlocal_nodes[i].u);
    ++num_nodes_for_pe[pe];
  });
  auto num_nodes_for_pe = num_nodes_for_pe_ets.combine(std::plus{});
  STOP_TIMER();

  SCOPED_TIMER("Exchange messages");
  return migrate_elements<GlobalNode>(num_nodes_for_pe, nonlocal_nodes, graph.communicator());
}

MigrationResult<GlobalEdge>
migrate_edges(const DistributedGraph &graph, const NoinitVector<GlobalEdge> &nonlocal_edges) {
  SCOPED_TIMER("Exchange nonlocal edges");

  const PEID size = mpi::get_comm_size(graph.communicator());

  START_TIMER("Barrier");
  MPI_Barrier(graph.communicator());
  STOP_TIMER();

  START_TIMER("Count messages");
  parallel::vector_ets<EdgeID> num_edges_for_pe_ets(size);
  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, nonlocal_edges.size()),
      [&](const auto &r) {
        auto &num_edges_for_pe = num_edges_for_pe_ets.local();
        PEID current_pe = 0;
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const GlobalNodeID u = nonlocal_edges[i].u;
          while (u >= graph.node_distribution(current_pe + 1)) {
            ++current_pe;
          }
          ++num_edges_for_pe[current_pe];
        }
      },
      tbb::static_partitioner{}
  );
  auto num_edges_for_pe = num_edges_for_pe_ets.combine(std::plus{});
  MPI_Barrier(graph.communicator());
  STOP_TIMER();

  SCOPED_TIMER("Exchange messages");
  return migrate_elements<GlobalEdge>(num_edges_for_pe, nonlocal_edges, graph.communicator());
}

MigratedNodesMapping exchange_migrated_nodes_mapping(
    const DistributedGraph &graph,
    const NoinitVector<GlobalNode> &nonlocal_nodes,
    const MigrationResult<GlobalNode> &local_nodes,
    const NoinitVector<NodeID> &lcluster_to_lcnode,
    const StaticArray<GlobalNodeID> &c_node_distribution
) {
  SCOPED_TIMER("Exchange node mapping for migrated nodes");

  const PEID rank = mpi::get_comm_rank(graph.communicator());

  NoinitVector<NodeMapping> their_nonlocal_to_gcnode(local_nodes.elements.size());
  NoinitVector<NodeID> their_req_to_lcnode(their_nonlocal_to_gcnode.size());

  tbb::parallel_for<std::size_t>(0, local_nodes.elements.size(), [&](const std::size_t i) {
    const GlobalNodeID gcluster = local_nodes.elements[i].u;
    const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
    const NodeID lcnode = lcluster_to_lcnode[lcluster];

    their_nonlocal_to_gcnode[i] = {
        .gcluster = gcluster,
        .gcnode = lcnode + c_node_distribution[rank],
    };
    their_req_to_lcnode[i] = lcnode;
  });

  NoinitVector<NodeMapping> my_nonlocal_to_gcnode(nonlocal_nodes.size());
  MPI_Alltoallv(
      their_nonlocal_to_gcnode.data(),
      local_nodes.recvcounts.data(),
      local_nodes.rdispls.data(),
      mpi::type::get<NodeMapping>(),
      my_nonlocal_to_gcnode.data(),
      local_nodes.sendcounts.data(),
      local_nodes.sdispls.data(),
      mpi::type::get<NodeMapping>(),
      graph.communicator()
  );

  return {std::move(my_nonlocal_to_gcnode), std::move(their_req_to_lcnode)};
}

template <typename T>
StaticArray<T> create_perfect_distribution_from_global_count(const T global_count, MPI_Comm comm) {
  const auto size = mpi::get_comm_size(comm);

  StaticArray<T> distribution(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    distribution[pe + 1] = math::compute_local_range<T>(global_count, size, pe).second;
  }

  return distribution;
}

template <bool migrate_prefix>
std::pair<NodeID, PEID> remap_gcnode(
    const GlobalNodeID gcnode,
    const PEID current_owner,
    const StaticArray<GlobalNodeID> &current_cnode_distribution,
    const NoinitVector<GlobalNodeID> &pe_overload,
    const NoinitVector<GlobalNodeID> &pe_underload
) {
  const auto lcnode = static_cast<NodeID>(gcnode - current_cnode_distribution[current_owner]);

  const auto old_current_owner_overload =
      static_cast<NodeID>(pe_overload[current_owner + 1] - pe_overload[current_owner]);
  const auto old_current_owner_count = static_cast<NodeID>(
      current_cnode_distribution[current_owner + 1] - current_cnode_distribution[current_owner]
  );
  const auto new_current_owner_count = old_current_owner_count - old_current_owner_overload;

  if constexpr (migrate_prefix) {
    // Remap the first few nodes to other PEs

    if (lcnode >= old_current_owner_overload) {
      return {lcnode - old_current_owner_overload, current_owner};
    } else {
      const GlobalNodeID position = pe_overload[current_owner] + lcnode;
      const PEID new_owner =
          static_cast<PEID>(math::find_in_distribution<GlobalNodeID>(position, pe_underload));
      const auto old_new_owner_count = static_cast<NodeID>(
          current_cnode_distribution[new_owner + 1] - current_cnode_distribution[new_owner]
      );
      const auto new_lcnode =
          static_cast<NodeID>(old_new_owner_count + position - pe_underload[new_owner]);
      return {new_lcnode, new_owner};
    }
  } else {
    // Remap the last few nodes to other PEs

    if (lcnode < new_current_owner_count) {
      return {lcnode, current_owner};
    } else {
      const GlobalNodeID position = pe_overload[current_owner] + lcnode - new_current_owner_count;
      const PEID new_owner =
          static_cast<PEID>(math::find_in_distribution<GlobalNodeID>(position, pe_underload));
      const auto old_new_owner_count = static_cast<NodeID>(
          current_cnode_distribution[new_owner + 1] - current_cnode_distribution[new_owner]
      );
      const auto new_lcnode =
          static_cast<NodeID>(old_new_owner_count + position - pe_underload[new_owner]);
      return {new_lcnode, new_owner};
    }
  }
}

AssignmentShifts compute_assignment_shifts(
    const StaticArray<GlobalNodeID> &current_node_distribution,
    const StaticArray<GlobalNodeID> &current_cnode_distribution,
    const double max_cnode_imbalance
) {
  const PEID size = static_cast<PEID>(current_cnode_distribution.size() - 1);
  const GlobalNodeID c_n = current_cnode_distribution.back();

  struct PELoad {
    PEID pe;
    GlobalNodeID count;
  };

  NoinitVector<PELoad> pe_load(size);
  NoinitVector<GlobalNodeID> pe_overload(size + 1);
  NoinitVector<GlobalNodeID> pe_underload(size + 1);

  pe_overload.front() = 0;
  pe_underload.front() = 0;

  const auto avg_cnode_count = static_cast<NodeID>(c_n / size);
  const auto max_cnode_count = static_cast<NodeID>(max_cnode_imbalance * avg_cnode_count);

  // Determine overloaded PEs
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    const auto cnode_count =
        static_cast<NodeID>(current_cnode_distribution[pe + 1] - current_cnode_distribution[pe]);
    pe_overload[pe + 1] = (cnode_count > max_cnode_count) ? cnode_count - max_cnode_count : 0;
    pe_load[pe] = {pe, cnode_count};
  });
  parallel::prefix_sum(pe_overload.begin(), pe_overload.end(), pe_overload.begin());
  const GlobalNodeID total_overload = pe_overload.back();

  // Sort PEs by load
  tbb::parallel_sort(pe_load.begin(), pe_load.end(), [&](const auto &lhs, const auto &rhs) {
    return lhs.count < rhs.count;
  });

  // Determine new load of underloaded PEs
  GlobalNodeID current_overload = total_overload;
  GlobalNodeID min_load = 0;
  PEID plus_ones = 0;
  PEID num_pes = 0;
  BinaryMinHeap<GlobalNodeID> maxes(size);

  for (PEID pe = 0; pe + 1 < size; ++pe) {
    const PEID actual_pe = pe_load[pe].pe;
    const GlobalNodeID pe_max =
        current_node_distribution[actual_pe + 1] - current_node_distribution[actual_pe];
    maxes.push(pe, pe_max); // id is never used
    ++num_pes;

    const GlobalNodeID prev_inc_from = pe_load[pe].count;
    GlobalNodeID inc_from = pe_load[pe].count;
    const GlobalNodeID inc_to = pe_load[pe + 1].count;

    while (!maxes.empty() && inc_to > maxes.peek_key()) {
      GlobalNodeID inc_to_prime = maxes.peek_key();
      GlobalNodeID delta = inc_to_prime - inc_from;
      maxes.pop();

      if (current_overload > num_pes * delta) {
        current_overload -= num_pes * delta;
        inc_from = inc_to_prime;
        --num_pes;
      } else {
        break;
      }
    }

    GlobalNodeID delta = inc_to - inc_from;

    if (current_overload > num_pes * delta) {
      current_overload -= num_pes * delta;
    } else {
      min_load = pe_load[pe].count + (inc_from - prev_inc_from) + current_overload / num_pes;
      plus_ones = current_overload % num_pes;
      current_overload = 0;
      break;
    }
  }

  if (current_overload > 0) {
    // Balancing clusters is not possible due to the constraint that no PE may gain more nodes
    // vertices than it has fine nodes (this is not an inherent constraint, but the remaining
    // coarsening codes requires is)
    // Hacky max_cnode_imbalance increase: @todo compute actual minimum achievable cnode imbalance
    // ...
    const double new_max_cnode_imbalance =
        1.01 * (max_cnode_count + current_overload) / avg_cnode_count;
    LOG_WARNING << "Cannot achieve maximum cnode imbalance: this should only ever happen in rare "
                   "edge cases; increasing maximum cnode imbalance constraint from "
                << max_cnode_imbalance << " to " << new_max_cnode_imbalance;
    return compute_assignment_shifts(
        current_node_distribution, current_cnode_distribution, new_max_cnode_imbalance
    );
  }

  // Determine underloaded PEs
  PEID nth_underloaded = 0;
  for (PEID pe = 0; pe < size; ++pe) {
    const auto cnode_count =
        static_cast<NodeID>(current_cnode_distribution[pe + 1] - current_cnode_distribution[pe]);

    if (cnode_count <= min_load) {
      const auto node_count =
          static_cast<NodeID>(current_node_distribution[pe + 1] - current_node_distribution[pe]);

      pe_underload[pe + 1] = std::min<NodeID>(min_load - cnode_count, node_count - cnode_count);
      if (nth_underloaded < plus_ones && pe_underload[pe + 1] < node_count - cnode_count) {
        ++pe_underload[pe + 1];
        ++nth_underloaded;
      }
    } else {
      pe_underload[pe + 1] = 0;
    }
  };
  parallel::prefix_sum(pe_underload.begin(), pe_underload.end(), pe_underload.begin());

  // If everything is correct, the total overload should match the total
  // underload
  KASSERT(
      [&] {
        if (pe_underload.back() != pe_overload.back()) {
          LOG_WARNING << V(pe_underload) << V(pe_overload) << V(total_overload)
                      << V(avg_cnode_count) << V(max_cnode_count) << V(min_load)
                      << V(current_node_distribution) << V(current_cnode_distribution);
          return false;
        }
        return true;
      }(),
      "",
      assert::light
  );

  return {
      .overload = std::move(pe_overload),
      .underload = std::move(pe_underload),
  };
}

void rebalance_cluster_placement(
    const DistributedGraph &graph,
    const StaticArray<GlobalNodeID> &current_cnode_distribution,
    const NoinitVector<NodeID> &lcluster_to_lcnode,
    const NoinitVector<NodeMapping> &nonlocal_gcluster_to_gcnode,
    GlobalClustering &lnode_to_gcluster,
    const double max_cnode_imbalance,
    const double migrate_cnode_prefix
) {
  SCOPED_TIMER("Rebalance cluster assignment");

  const auto shifts = compute_assignment_shifts(
      graph.node_distribution(), current_cnode_distribution, max_cnode_imbalance
  );

  // Now remap the cluster IDs such that we respect pe_overload and pe_overload
  growt::GlobalNodeIDMap<GlobalNodeID> nonlocal_gcluster_to_gcnode_map(
      nonlocal_gcluster_to_gcnode.size()
  );
  tbb::enumerable_thread_specific<growt::GlobalNodeIDMap<GlobalNodeID>::handle_type>
      nonlocal_gcluster_to_gcnode_handle_ets([&] {
        return nonlocal_gcluster_to_gcnode_map.get_handle();
      });

  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, nonlocal_gcluster_to_gcnode.size()),
      [&](const auto &r) {
        auto &handle = nonlocal_gcluster_to_gcnode_handle_ets.local();
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const auto &[gcluster, gcnode] = nonlocal_gcluster_to_gcnode[i];
          handle.insert(gcluster + 1, gcnode);
        }
      }
  );

  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  graph.pfor_nodes_range([&](const auto &r) {
    auto &handle = nonlocal_gcluster_to_gcnode_handle_ets.local();

    for (NodeID lnode = r.begin(); lnode != r.end(); ++lnode) {
      const GlobalNodeID old_gcluster = lnode_to_gcluster[lnode];

      GlobalNodeID old_gcnode = 0;
      PEID old_owner = 0;
      if (graph.is_owned_global_node(old_gcluster)) {
        const NodeID old_lcluster = static_cast<NodeID>(old_gcluster - graph.offset_n());
        old_gcnode = lcluster_to_lcnode[old_lcluster] + current_cnode_distribution[rank];
        old_owner = rank;
      } else {
        auto it = handle.find(old_gcluster + 1);
        KASSERT(it != handle.end());
        old_gcnode = (*it).second;
        old_owner = graph.find_owner_of_global_node(old_gcluster);
      }

      const auto [new_lcnode, new_owner] = [&] {
        if (migrate_cnode_prefix) {
          return remap_gcnode<true>(
              old_gcnode, old_owner, current_cnode_distribution, shifts.overload, shifts.underload
          );
        } else {
          return remap_gcnode<false>(
              old_gcnode, old_owner, current_cnode_distribution, shifts.overload, shifts.underload
          );
        }
      }();
      KASSERT(new_owner < size);

      lnode_to_gcluster[lnode] = graph.offset_n(new_owner) + new_lcnode;

      // We have to ensure that coarse node IDs assigned to this PE are still within the range of
      // the fine node IDs assigned to this PEs, otherwise many implicit assertions that follow
      // are violated
      KASSERT(lnode_to_gcluster[lnode] >= graph.offset_n(new_owner));
      KASSERT(lnode_to_gcluster[lnode] < graph.offset_n(new_owner + 1));
    }
  });

  // Synchronize the new labels of ghost nodes
  struct Message {
    NodeID lnode;
    GlobalNodeID gcluster;
  };
  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID lnode) -> Message {
        return {lnode, lnode_to_gcluster[lnode]};
      },
      [&](const auto buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &[their_lnode, new_gcluster] = buffer[i];

          const NodeID lnode = graph.global_to_local_node(graph.offset_n(pe) + their_lnode);
          lnode_to_gcluster[lnode] = new_gcluster;
        });
      }
  );
}
} // namespace

namespace debug {
bool validate_clustering(const DistributedGraph &graph, const GlobalClustering &lnode_to_gcluster) {
  for (const NodeID lnode : graph.all_nodes()) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
    if (gcluster > graph.global_n()) {
      LOG_WARNING << "Invalid clustering for local node " << lnode << ": " << gcluster
                  << "; aborting";
      return false;
    }
  }

  struct Message {
    NodeID lnode;
    GlobalNodeID gcluster;
  };

  std::atomic<std::uint8_t> failed = false;
  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID u) -> Message { return {.lnode = u, .gcluster = lnode_to_gcluster[u]}; },
      [&](const auto &recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          if (failed) {
            return;
          }

          const auto [their_lnode, gcluster] = recv_buffer[i];
          const auto gnode = static_cast<GlobalNodeID>(graph.offset_n(pe) + their_lnode);
          const NodeID lnode = graph.global_to_local_node(gnode);
          if (lnode_to_gcluster[lnode] != gcluster) {
            LOG_WARNING << "Inconsistent cluster for local node " << lnode
                        << " (ghost node, global node ID " << gnode << "): "
                        << "the node is owned by PE " << pe
                        << ", which assigned the node to cluster " << gcluster
                        << ", but our ghost node is assigned to cluster "
                        << lnode_to_gcluster[lnode] << "; aborting";
            failed = 1;
          }
        });
      }
  );
  return failed == 0;
}
} // namespace debug

ContractionResult contract_clustering(
    const DistributedGraph &graph,
    GlobalClustering &lnode_to_gcluster,
    const double max_cnode_imbalance,
    const bool migrate_cnode_prefix,
    const bool force_perfect_cnode_balance
) {
  TIMER_BARRIER(graph.communicator());
  START_TIMER("Contract clustering");

  KASSERT(
      debug::validate_clustering(graph, lnode_to_gcluster),
      "input clustering is invalid",
      assert::heavy
  );

  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  // Collect nodes and edges that must be migrated to another PE:
  // - nodes that are assigned to non-local clusters
  // - edges whose source is a node in a non-local cluster
  auto nonlocal_nodes = find_nonlocal_nodes(graph, lnode_to_gcluster);

  IF_STATS {
    const auto total_num_nonlocal_nodes =
        mpi::allreduce<GlobalNodeID>(nonlocal_nodes.size(), MPI_SUM, graph.communicator());
    STATS << "Total number of nonlocal nodes: " << total_num_nonlocal_nodes;
  }

  sort_node_list(nonlocal_nodes);

  // Migrate those nodes and edges
  auto migration_result_nodes = migrate_nodes(graph, nonlocal_nodes);
  auto &local_nodes = migration_result_nodes.elements;

  // Map non-empty clusters belonging to this PE to a consecutive range of
  // coarse node IDs:
  // ```
  // lnode_to_lcnode[local node ID] = local coarse node ID
  // ```
  auto lcluster_to_lcnode = build_lcluster_to_lcnode_mapping(graph, lnode_to_gcluster, local_nodes);

  // Make cluster IDs start at 0
  NodeID c_n = lcluster_to_lcnode.empty() ? 0 : lcluster_to_lcnode.back() + 1;
  auto c_node_distribution = build_distribution<GlobalNodeID>(c_n, graph.communicator());
  DBG << "Coarse node distribution: [" << c_node_distribution << "]";

  // To construct the mapping[] array, we need to know the mapping of nodes that
  // we migrated to another PE to coarse node IDs: exchange these mappings here
  auto mapping_exchange_result = exchange_migrated_nodes_mapping(
      graph, nonlocal_nodes, migration_result_nodes, lcluster_to_lcnode, c_node_distribution
  );

  // Mapping from local nodes that belong to non-local clusters to coarse nodes:
  // .gcluster -- global cluster that belongs to another PE (but have at least
  // one node in this cluster) .gcnode -- the corresponding coarse node (global
  // ID)
  auto &my_nonlocal_to_gcnode = mapping_exchange_result.my_nonlocal_to_gcnode;

  // Mapping from node migration messages that we received (i.e., messages that
  // other PEs send us since they own nodes that belong to some cluster owned by
  // our PE) to the corresponding coarse node (local ID) We don't need this
  // information during contraction, but can use it to send the block assignment
  // of coarse nodes to other PEs during uncoarsening
  auto &their_req_to_lcnode = mapping_exchange_result.their_req_to_lcnode;

  // If the "natural" assignment of coarse nodes to PEs has too much imbalance,
  // we remap the cluster IDs to achieve perfect coarse node balance
  if (const double cnode_imbalance = compute_distribution_imbalance(c_node_distribution);
      cnode_imbalance > max_cnode_imbalance) {
    STATS << "Natural cluster assignment exceeds maximum coarse node imbalance: " << cnode_imbalance
          << " > " << max_cnode_imbalance << " --> rebalancing cluster assignment";

    const double max_imbalance = force_perfect_cnode_balance ? 1.0 : max_cnode_imbalance;
    rebalance_cluster_placement(
        graph,
        c_node_distribution,
        lcluster_to_lcnode,
        my_nonlocal_to_gcnode,
        lnode_to_gcluster,
        max_imbalance,
        migrate_cnode_prefix
    );

    STOP_TIMER(); // Contract clustering timer

    // In some edge cases, rebalance_cluster_placement() might fail to balance the clusters to
    // max_imbalance (this is because the subgraph of a PE cannot grow in size during coarsening).
    // Thus, we accept any imbalance for the "rebalanced try" to avoid an infinite loop.
    // @todo can this actually happen?
    return contract_clustering(graph, lnode_to_gcluster);
  }

  auto nonlocal_edges = find_nonlocal_edges(graph, lnode_to_gcluster);

  IF_STATS {
    const auto total_num_nonlocal_edges =
        mpi::allreduce<GlobalEdgeID>(nonlocal_edges.size(), MPI_SUM, graph.communicator());
    const auto max_num_nonlocal_edges =
        mpi::allreduce<GlobalEdgeID>(nonlocal_edges.size(), MPI_MAX, graph.communicator());
    STATS << "Total number of nonlocal edges before deduplication: " << total_num_nonlocal_edges
          << "; max: " << max_num_nonlocal_edges
          << "; imbalance: " << max_num_nonlocal_edges / (1.0 * total_num_nonlocal_edges / size);
  }

  deduplicate_edge_list(nonlocal_edges);

  IF_STATS {
    const auto total_num_nonlocal_edges =
        mpi::allreduce<GlobalEdgeID>(nonlocal_edges.size(), MPI_SUM, graph.communicator());
    const auto max_num_nonlocal_edges =
        mpi::allreduce<GlobalEdgeID>(nonlocal_edges.size(), MPI_MAX, graph.communicator());
    STATS << "Total number of nonlocal edges after deduplication: " << total_num_nonlocal_edges
          << "; max: " << max_num_nonlocal_edges
          << "; imbalance: " << max_num_nonlocal_edges / (1.0 * total_num_nonlocal_edges / size);
  }

  auto migration_result_edges = migrate_edges(graph, nonlocal_edges);
  auto &local_edges = migration_result_edges.elements;

  START_TIMER("Sort migrated local edges");
  tbb::parallel_sort(local_edges.begin(), local_edges.end(), [&](const auto &lhs, const auto &rhs) {
    return lhs.u < rhs.u;
  });
  STOP_TIMER();

  // Next, exchange the mapping of ghost nodes to coarse nodes
  START_TIMER("Communicate mapping for ghost nodes");
  using NonlocalClusterMap = growt::StaticGhostNodeMapping;
  NonlocalClusterMap nonlocal_gcluster_to_index(graph.total_n() + local_edges.size());
  std::vector<parallel::Aligned<parallel::Atomic<NodeID>>> next_index_for_pe(size + 1);

  auto request_nonlocal_mapping = [&](const GlobalNodeID cluster) {
    const auto [it, mine] = nonlocal_gcluster_to_index.insert(cluster + 1, 1); // dummy value
    if (mine) {
      const PEID owner = graph.find_owner_of_global_node(cluster);
      nonlocal_gcluster_to_index.update(cluster + 1, [&](auto &lhs) {
        return lhs = ++(next_index_for_pe[owner].value);
      });
    }
  };

  tbb::parallel_invoke(
      [&] {
        graph.pfor_nodes([&](const NodeID u) {
          const GlobalNodeID gcluster_u = lnode_to_gcluster[u];
          if (!graph.is_owned_global_node(gcluster_u)) {
            return;
          }

          for (const auto [e, v] : graph.neighbors(u)) {
            const GlobalNodeID gcluster_v = lnode_to_gcluster[v];
            if (!graph.is_owned_global_node(gcluster_v)) {
              request_nonlocal_mapping(gcluster_v);
            }
          }
        });
      },
      [&] {
        tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
          const GlobalNodeID gcluster_v = local_edges[i].v;
          if (!graph.is_owned_global_node(gcluster_v)) {
            request_nonlocal_mapping(gcluster_v);
          }
        });
      }
  );

  // @bug not all entries are used, why?
  std::vector<scalable_vector<GlobalNodeID>> my_mapping_requests(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    my_mapping_requests[pe].resize(next_index_for_pe[pe].value, kInvalidGlobalNodeID);
  });

  growt::pfor_map(
      nonlocal_gcluster_to_index,
      [&](const GlobalNodeID gcluster_p1, const std::size_t index_p1) {
        const GlobalNodeID gcluster = gcluster_p1 - 1;
        const std::size_t index = index_p1 - 1;
        const PEID owner = graph.find_owner_of_global_node(gcluster);

        KASSERT(owner < static_cast<PEID>(my_mapping_requests.size()));
        KASSERT(index < my_mapping_requests[owner].size());
        KASSERT(gcluster != 0u || owner == 0);

        my_mapping_requests[owner][index] = gcluster;
      }
  );

  auto their_mapping_requests =
      mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_requests, graph.communicator());

  std::vector<scalable_vector<GlobalNodeID>> my_mapping_responses(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    my_mapping_responses[pe].resize(their_mapping_requests[pe].size());

    tbb::parallel_for<std::size_t>(0, their_mapping_requests[pe].size(), [&](const std::size_t i) {
      KASSERT(static_cast<std::size_t>(pe) < their_mapping_requests.size());
      KASSERT(i < their_mapping_requests[pe].size());
      const GlobalNodeID global = their_mapping_requests[pe][i];
      KASSERT(global != kInvalidGlobalNodeID);
      KASSERT(global >= graph.offset_n(), V(rank));
      const auto local = static_cast<NodeID>(global - graph.offset_n());
      KASSERT(local < lcluster_to_lcnode.size());
      const NodeID coarse_local = lcluster_to_lcnode[local];
      KASSERT(static_cast<std::size_t>(rank) < c_node_distribution.size());
      const GlobalNodeID coarse_global = c_node_distribution[rank] + coarse_local;
      KASSERT(static_cast<std::size_t>(pe) < my_mapping_responses.size());
      KASSERT(i < my_mapping_responses[pe].size());
      my_mapping_responses[pe][i] = coarse_global;
    });
  });

  auto their_mapping_responses =
      mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_responses, graph.communicator());
  STOP_TIMER();

  // Build the coarse ghost node mapping: coarse ghost nodes to coarse global
  // nodes
  START_TIMER("Build mapping");
  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, my_nonlocal_to_gcnode.size()),
      [&](const auto &r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const auto &[local_cluster, global_coarse_node] = my_nonlocal_to_gcnode[i];
          nonlocal_gcluster_to_index.insert(
              local_cluster + 1, graph.global_n() + global_coarse_node + 1
          );
        }
      }
  );

  // Build a mapping array from fine nodes to coarse nodes
  NoinitVector<GlobalNodeID> lnode_to_gcnode(graph.n());
  graph.pfor_nodes([&](const NodeID u) {
    const GlobalNodeID cluster = lnode_to_gcluster[u];

    if (graph.is_owned_global_node(cluster)) {
      lnode_to_gcnode[u] =
          lcluster_to_lcnode[cluster - graph.offset_n()] + c_node_distribution[rank];
    } else {
      auto it = nonlocal_gcluster_to_index.find(cluster + 1);
      KASSERT(it != nonlocal_gcluster_to_index.end());

      const std::size_t index = (*it).second - 1;
      if (index < graph.global_n()) {
        const PEID owner = graph.find_owner_of_global_node(cluster);
        lnode_to_gcnode[u] = their_mapping_responses[owner][index];
      } else {
        lnode_to_gcnode[u] = static_cast<GlobalNodeID>(index - graph.global_n());
      }
    }

    KASSERT(lnode_to_gcnode[u] < c_node_distribution.back());
  });

  // Build mapping for ghost nodes
  std::exclusive_scan(
      next_index_for_pe.begin(),
      next_index_for_pe.end(),
      next_index_for_pe.begin(),
      parallel::Aligned<parallel::Atomic<NodeID>>(0),
      [](const auto &init, const auto &op) {
        return parallel::Aligned<parallel::Atomic<NodeID>>(init.value + op.value);
      }
  );

  const NodeID c_ghost_n = next_index_for_pe.back().value;
  growt::StaticGhostNodeMapping c_global_to_ghost(c_ghost_n);
  StaticArray<GlobalNodeID> c_ghost_to_global(c_ghost_n);
  StaticArray<PEID> c_ghost_owner(c_ghost_n);

  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    for (std::size_t i = 0; i < my_mapping_requests[pe].size(); ++i) {
      const GlobalNodeID global = their_mapping_responses[pe][i];
      const NodeID local = next_index_for_pe[pe].value + i;
      c_global_to_ghost.insert(global + 1, c_n + local);
      c_ghost_to_global[local] = global;
      c_ghost_owner[local] = pe;
    }
  });
  STOP_TIMER();

  //
  // Sort local nodes by their cluster ID
  //
  localize_global_edge_list(local_edges, graph.offset_n(), lcluster_to_lcnode);

  auto bucket_sort_result =
      build_node_buckets(graph, lcluster_to_lcnode, c_n, local_edges, lnode_to_gcluster);
  auto &buckets_position_buffer = bucket_sort_result.first;
  auto &buckets = bucket_sort_result.second;

  //
  // Construct the coarse edges
  //
  START_TIMER("Allocation");
  StaticArray<EdgeID> c_nodes(c_n + 1);
  StaticArray<NodeWeight> c_node_weights(c_n + c_ghost_n);
  STOP_TIMER();

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector_ets([&] {
    return RatingMap<EdgeWeight, NodeID>(c_n + c_ghost_n);
  });

  struct LocalEdge {
    NodeID node;
    EdgeWeight weight;
  };

  NavigableLinkedList<NodeID, LocalEdge, scalable_vector> edge_buffer_ets;

  START_TIMER("Construct edges");
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
    auto &collector = collector_ets.local();
    auto &edge_buffer = edge_buffer_ets.local();

    for (NodeID lcu = r.begin(); lcu != r.end(); ++lcu) {
      edge_buffer.mark(lcu);

      const std::size_t first_pos = buckets_position_buffer[lcu];
      const std::size_t last_pos = buckets_position_buffer[lcu + 1];

      auto collect_edges = [&](auto &map) {
        NodeWeight c_u_weight = 0;

        for (std::size_t i = first_pos; i < last_pos; ++i) {
          const NodeID u = buckets[i];

          auto handle_edge_to_gcluster = [&](const EdgeWeight weight, const GlobalNodeID gcluster) {
            if (graph.is_owned_global_node(gcluster)) {
              const auto lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
              const auto lcnode = lcluster_to_lcnode[lcluster];
              if (lcnode != lcu) {
                map[lcnode] += weight;
              }
            } else {
              auto it = nonlocal_gcluster_to_index.find(gcluster + 1);
              KASSERT(
                  it != nonlocal_gcluster_to_index.end() && (*it).second - 1 < graph.global_n()
              );

              const std::size_t index = (*it).second - 1;
              const PEID owner = graph.find_owner_of_global_node(gcluster);
              const GlobalNodeID gcnode = their_mapping_responses[owner][index];
              auto lcnode_it = c_global_to_ghost.find(gcnode + 1);
              const NodeID lcnode = (*lcnode_it).second;

              if (lcnode != lcu) {
                map[lcnode] += weight;
              }
            }
          };

          auto handle_edge_to_lnode = [&](const EdgeWeight weight, const NodeID lnode) {
            if (graph.is_owned_node(lnode)) {
              const GlobalNodeID gcnode = lnode_to_gcnode[lnode];
              const bool is_local_gcnode =
                  (gcnode >= c_node_distribution[rank] && gcnode < c_node_distribution[rank + 1]);

              if (is_local_gcnode) {
                const auto lcnode = static_cast<NodeID>(gcnode - c_node_distribution[rank]);
                if (lcu != lcnode) {
                  map[lcnode] += weight;
                }
              } else {
                auto lcnode_it = c_global_to_ghost.find(gcnode + 1);
                KASSERT(lcnode_it != c_global_to_ghost.end());
                const NodeID lcnode = (*lcnode_it).second;
                if (lcu != lcnode) {
                  map[lcnode] += weight;
                }
              }
            } else {
              handle_edge_to_gcluster(weight, lnode_to_gcluster[lnode]);
            }
          };

          if (u < graph.n()) {
            c_u_weight += graph.node_weight(u);
            for (const auto [e, v] : graph.neighbors(u)) {
              handle_edge_to_lnode(graph.edge_weight(e), v);
            }
          } else {
            // Fix node weight later
            for (std::size_t index = u - graph.n();
                 index < local_edges.size() && local_edges[index].u == lcu;
                 ++index) {
              handle_edge_to_gcluster(local_edges[index].weight, local_edges[index].v);
            }
          }
        }

        c_node_weights[lcu] = c_u_weight;
        c_nodes[lcu + 1] = map.size();

        for (const auto [c_v, weight] : map.entries()) {
          edge_buffer.push_back({c_v, weight});
        }
        map.clear();
      };

      EdgeID upper_bound_degree = 0;
      for (std::size_t i = first_pos; i < last_pos; ++i) {
        const NodeID u = buckets[i];
        if (u < graph.n()) {
          upper_bound_degree += graph.degree(u);
        } else {
          for (std::size_t index = u - graph.n();
               index < local_edges.size() && local_edges[index].u == lcu;
               ++index) {
            ++upper_bound_degree;
          }
        }
      }
      upper_bound_degree = static_cast<EdgeID>(
          std::min<GlobalNodeID>(c_n + c_ghost_n, static_cast<GlobalNodeID>(upper_bound_degree))
      );
      collector.execute(upper_bound_degree, collect_edges);
    }
  });

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());
  STOP_TIMER();

  START_TIMER("Integrate node weights of migrated nodes");
  tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
    const NodeID c_u = lcluster_to_lcnode[local_nodes[i].u - graph.offset_n()];
    __atomic_fetch_add(&c_node_weights[c_u], local_nodes[i].weight, __ATOMIC_RELAXED);
  });
  STOP_TIMER();

  // Build edge distribution
  const EdgeID c_m = c_nodes.back();
  auto c_edge_distribution = build_distribution<GlobalEdgeID>(c_m, graph.communicator());
  DBG << "Coarse edge distribution: [" << c_edge_distribution << "]";

  START_TIMER("Allocation");
  StaticArray<NodeID> c_edges(c_m);
  StaticArray<EdgeWeight> c_edge_weights(c_m);
  STOP_TIMER();

  // Finally, build coarse graph
  START_TIMER("Construct coarse graph");
  auto all_buffered_nodes =
      ts_navigable_list::combine<NodeID, LocalEdge, scalable_vector>(edge_buffer_ets);

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

  DistributedGraph c_graph(
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
  );
  STOP_TIMER();

  update_ghost_node_weights(c_graph);

  STOP_TIMER(); // Contract clustering timer

  return {
      std::move(c_graph),
      std::move(lnode_to_gcnode),
      {
          .nodes = std::move(their_req_to_lcnode),
          .sendcounts = std::move(migration_result_nodes.recvcounts),
          .sdispls = std::move(migration_result_nodes.rdispls),
          .recvcounts = std::move(migration_result_nodes.sendcounts),
          .rdispls = std::move(migration_result_nodes.sdispls),
      },
  };
}

DistributedPartitionedGraph project_partition(
    const DistributedGraph &graph,
    DistributedPartitionedGraph p_c_graph,
    const NoinitVector<GlobalNodeID> &c_mapping,
    const MigratedNodes &migration
) {
  SCOPED_TIMER("Project partition");

  struct MigratedNodeBlock {
    GlobalNodeID gcnode;
    BlockID block;
  };
  NoinitVector<MigratedNodeBlock> migrated_nodes_sendbuf(
      migration.sdispls.back() + migration.sendcounts.back()
  );
  NoinitVector<MigratedNodeBlock> migrated_nodes_recvbuf(
      migration.rdispls.back() + migration.recvcounts.back()
  );

  TIMED_SCOPE("Exchange migrated node blocks") {
    tbb::parallel_for<std::size_t>(0, migrated_nodes_sendbuf.size(), [&](const std::size_t i) {
      const NodeID lcnode = migration.nodes[i];
      const BlockID block = p_c_graph.block(lcnode);
      const GlobalNodeID gcnode = p_c_graph.local_to_global_node(lcnode);
      migrated_nodes_sendbuf[i] = {.gcnode = gcnode, .block = block};
    });

    MPI_Alltoallv(
        migrated_nodes_sendbuf.data(),
        migration.sendcounts.data(),
        migration.sdispls.data(),
        mpi::type::get<MigratedNodeBlock>(),
        migrated_nodes_recvbuf.data(),
        migration.recvcounts.data(),
        migration.rdispls.data(),
        mpi::type::get<MigratedNodeBlock>(),
        graph.communicator()
    );
  };

  START_TIMER("Allocation");
  StaticArray<BlockID> partition(graph.total_n());
  STOP_TIMER();

  TIMED_SCOPE("Building projected partition array") {
    growt::GlobalNodeIDMap<GlobalNodeID> gcnode_to_block(0);
    tbb::enumerable_thread_specific<growt::GlobalNodeIDMap<GlobalNodeID>::handle_type>
        gcnode_to_block_handle_ets([&] { return gcnode_to_block.get_handle(); });
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, migrated_nodes_recvbuf.size()),
        [&](const auto &r) {
          auto &gcnode_to_block_handle = gcnode_to_block_handle_ets.local();
          for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto &migrated_node = migrated_nodes_recvbuf[i];
            gcnode_to_block_handle.insert(migrated_node.gcnode + 1, migrated_node.block);
          }
        }
    );

    graph.pfor_nodes_range([&](const auto &r) {
      auto &gcnode_to_block_handle = gcnode_to_block_handle_ets.local();

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const GlobalNodeID gcnode = c_mapping[u];
        if (p_c_graph.is_owned_global_node(gcnode)) {
          const NodeID lcnode = p_c_graph.global_to_local_node(gcnode);
          partition[u] = p_c_graph.block(lcnode);
        } else {
          auto it = gcnode_to_block_handle.find(gcnode + 1);
          KASSERT(it != gcnode_to_block_handle.end(), V(gcnode));
          partition[u] = (*it).second;
        }
      }
    });
  };

  struct GhostNodeLabel {
    NodeID local_node_on_sender;
    BlockID block;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeLabel>(
      graph,
      [&](const NodeID lnode) -> GhostNodeLabel {
        return {lnode, partition[lnode]};
      },
      [&](const auto buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &[sender_lnode, block] = buffer[i];
          const GlobalNodeID gnode = graph.offset_n(pe) + sender_lnode;
          const NodeID lnode = graph.global_to_local_node(gnode);
          partition[lnode] = block;
        });
      }
  );

  return {&graph, p_c_graph.k(), std::move(partition), p_c_graph.take_block_weights()};
}
} // namespace kaminpar::dist
