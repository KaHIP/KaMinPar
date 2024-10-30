/*******************************************************************************
 * Graph contraction for arbitrary clusterings.
 *
 * In this file, we use the following naming sheme for node and cluster IDs:
 * - {g,l}[c]{node,cluster}
 *    ^ global or local ID
 *         ^ ID in [c]oarse graph or in fine graph
 *            ^ node or cluster ID
 *
 * @file:   global_cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 * @brief:  Graph contraction for arbitrary clusterings.
 ******************************************************************************/
#include "kaminpar-dist/coarsening/contraction/global_cluster_contraction.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>

#include "kaminpar-dist/coarsening/contraction.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/vector_ets.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {

namespace {

SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

} // namespace

namespace {

// Stores technical mappings necessary to project a partition of the coarse graph to the fine graph.
// Part of the contraction result and should not be used outside the `project_partition()` function.
struct MigratedNodes {
  StaticArray<NodeID> nodes;

  std::vector<int> sendcounts;
  std::vector<int> sdispls;
  std::vector<int> recvcounts;
  std::vector<int> rdispls;
};

class GlobalCoarseGraphImpl : public CoarseGraph {
public:
  GlobalCoarseGraphImpl(
      const DistributedGraph &f_graph,
      DistributedGraph c_graph,
      StaticArray<GlobalNodeID> mapping,
      MigratedNodes migration
  )
      : _f_graph(f_graph),
        _c_graph(std::move(c_graph)),
        _lnode_to_gcnode(std::move(mapping)),
        _migration(std::move(migration)) {}

  const DistributedGraph &get() const final {
    return _c_graph;
  }

  DistributedGraph &get() final {
    return _c_graph;
  }

  void project(const StaticArray<BlockID> &c_partition, StaticArray<BlockID> &f_partition) final {
    SCOPED_TIMER("Project partition");
    SCOPED_HEAP_PROFILER("Project partition");

    struct MigratedNodeBlock {
      GlobalNodeID gcnode;
      BlockID block;
    };
    StaticArray<MigratedNodeBlock> migrated_nodes_sendbuf(
        _migration.sdispls.back() + _migration.sendcounts.back()
    );
    StaticArray<MigratedNodeBlock> migrated_nodes_recvbuf(
        _migration.rdispls.back() + _migration.recvcounts.back()
    );

    TIMED_SCOPE("Exchange migrated node blocks") {
      _c_graph.reified([&](const auto &graph) {
        tbb::parallel_for<std::size_t>(0, migrated_nodes_sendbuf.size(), [&](const std::size_t i) {
          const NodeID lcnode = _migration.nodes[i];
          const BlockID block = c_partition[lcnode];
          const GlobalNodeID gcnode = graph.local_to_global_node(lcnode);
          migrated_nodes_sendbuf[i] = {.gcnode = gcnode, .block = block};
        });
      });

      MPI_Alltoallv(
          migrated_nodes_sendbuf.data(),
          _migration.sendcounts.data(),
          _migration.sdispls.data(),
          mpi::type::get<MigratedNodeBlock>(),
          migrated_nodes_recvbuf.data(),
          _migration.recvcounts.data(),
          _migration.rdispls.data(),
          mpi::type::get<MigratedNodeBlock>(),
          _f_graph.communicator()
      );
    };

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

      _c_graph.reified([&](const auto &graph) {
        _f_graph.pfor_all_nodes([&](const NodeID lnode) {
          const GlobalNodeID gcnode = _lnode_to_gcnode[lnode];
          if (graph.is_owned_global_node(gcnode)) {
            const NodeID lcnode = graph.global_to_local_node(gcnode);
            f_partition[lnode] = c_partition[lcnode];
          } else {
            auto &gcnode_to_block_handle = gcnode_to_block_handle_ets.local();
            auto it = gcnode_to_block_handle.find(gcnode + 1);
            KASSERT(it != gcnode_to_block_handle.end(), V(gcnode));
            f_partition[lnode] = (*it).second;
          }
        });
      });
    };
  }

private:
  const DistributedGraph &_f_graph;
  DistributedGraph _c_graph;
  StaticArray<GlobalNodeID> _lnode_to_gcnode;
  MigratedNodes _migration;
};
} // namespace

namespace {
struct AssignmentShifts {
  StaticArray<GlobalNodeID> overload;
  StaticArray<GlobalNodeID> underload;
};

struct GlobalEdge {
  GlobalNodeID u;
  GlobalNodeID v;
  EdgeWeight weight;
};

struct GlobalNode {
  GlobalNodeID gcluster;
  NodeWeight weight;
};

template <typename T> struct MigrationResult {
  StaticArray<T> elements;

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
  StaticArray<NodeMapping> my_nonlocal_to_gcnode;
  StaticArray<NodeID> their_req_to_lcnode;
};

struct ClusterToCoarseHandle {
  ClusterToCoarseHandle(
      const PEID rank,
      const StaticArray<GlobalNodeID> &node_distribution,
      const StaticArray<GlobalNodeID> &c_node_distribution,
      const StaticArray<NodeID> &lcluster_to_lcnode,
      growt::GlobalNodeIDMap<GlobalNodeID>::handle_type nonlocal_gcluster_to_gcnode
  )
      : _rank(rank),
        _from(node_distribution[rank]),
        _to(node_distribution[rank + 1]),
        _c_from(c_node_distribution[rank]),
        _lcluster_to_lcnode(lcluster_to_lcnode),
        _nonlocal_gcluster_to_gcnode(std::move(nonlocal_gcluster_to_gcnode)) {}

  GlobalNodeID nonlocal_gcluster_to_gcnode(const GlobalNodeID gcluster) {
    auto it = _nonlocal_gcluster_to_gcnode.find(gcluster + 1);
    KASSERT(it != _nonlocal_gcluster_to_gcnode.end());
    return (*it).second;
  }

  GlobalNodeID local_gcluster_to_gcnode(const GlobalNodeID gcluster) {
    const NodeID lcluster = static_cast<NodeID>(gcluster - _from);
    const NodeID lcnode = _lcluster_to_lcnode[lcluster];
    return _c_from + lcnode;
  }

  GlobalNodeID gcluster_to_gcnode(const GlobalNodeID gcluster) {
    if (_from <= gcluster && gcluster < _to) {
      return local_gcluster_to_gcnode(gcluster);
    } else {
      return nonlocal_gcluster_to_gcnode(gcluster);
    }
  }

  NodeID lcluster_to_lcnode(const NodeID lcluster) {
    KASSERT(lcluster < _lcluster_to_lcnode.size());
    return _lcluster_to_lcnode[lcluster];
  }

  PEID _rank;
  GlobalNodeID _from;
  GlobalNodeID _to;
  GlobalNodeID _c_from;

  const StaticArray<NodeID> &_lcluster_to_lcnode;
  growt::GlobalNodeIDMap<GlobalNodeID>::handle_type _nonlocal_gcluster_to_gcnode;
};

class ClusterToCoarseMapper {
public:
  ClusterToCoarseMapper(
      const PEID rank,
      const StaticArray<GlobalNodeID> &node_distribution,
      const StaticArray<GlobalNodeID> &c_node_distribution,
      const StaticArray<NodeID> &lcluster_to_lcnode,
      const StaticArray<NodeMapping> &nonlocal_gcluster_to_gcnode_vec
  )
      : _rank(rank),
        _node_distribution(node_distribution),
        _c_node_distribution(c_node_distribution),
        _lcluster_to_lcnode(lcluster_to_lcnode),
        _nonlocal_gcluster_to_gcnode(nonlocal_gcluster_to_gcnode_vec.size()) {
    initialize_nonlocal_gcluster_to_gcnode(nonlocal_gcluster_to_gcnode_vec);
  }

  ClusterToCoarseHandle &handle() {
    return _handle_ets.local();
  }

private:
  void initialize_nonlocal_gcluster_to_gcnode(
      const StaticArray<NodeMapping> &nonlocal_gcluster_to_gcnode_vec
  ) {
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, nonlocal_gcluster_to_gcnode_vec.size()),
        [&](const auto &r) {
          auto &handle = _handle_ets.local()._nonlocal_gcluster_to_gcnode;

          for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto &[gcluster, gcnode] = nonlocal_gcluster_to_gcnode_vec[i];
            handle.insert(gcluster + 1, gcnode);
          }
        }
    );
  }

  PEID _rank;

  const StaticArray<GlobalNodeID> &_node_distribution;
  const StaticArray<GlobalNodeID> &_c_node_distribution;

  const StaticArray<NodeID> &_lcluster_to_lcnode;
  growt::GlobalNodeIDMap<GlobalNodeID> _nonlocal_gcluster_to_gcnode;

  tbb::enumerable_thread_specific<ClusterToCoarseHandle> _handle_ets{[&] {
    return ClusterToCoarseHandle{
        _rank,
        _node_distribution,
        _c_node_distribution,
        _lcluster_to_lcnode,
        _nonlocal_gcluster_to_gcnode.get_handle(),
    };
  }};
};

template <typename Graph>
StaticArray<GlobalNode>
find_nonlocal_nodes(const Graph &graph, const StaticArray<GlobalNodeID> &lnode_to_gcluster) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Collect nonlocal nodes");
  SCOPED_HEAP_PROFILER("Collect nonlocal nodes");

  // @TODO parallelize

  // growt::StaticGhostNodeMapping nonlocal_nodes(graph.total_n());
  std::unordered_map<GlobalNodeID, NodeWeight> nonlocal_nodes;
  std::atomic<std::size_t> size = 0;

  for (NodeID lnode : graph.all_nodes()) {
    // graph.pfor_all_nodes([&](const NodeID lnode) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];

    if (graph.is_owned_global_node(gcluster)) {
      // return;
      continue;
    }

    const NodeWeight weight = graph.is_owned_node(lnode) ? graph.node_weight(lnode) : 0;

    // auto ans = nonlocal_nodes.insert_or_update(
    //     gcluster + 1, weight, [](auto &lhs, auto rhs) { return lhs = lhs + rhs; }, weight
    //);
    // if (ans.second) {
    //   size.fetch_add(1, std::memory_order_relaxed);
    // }

    if (nonlocal_nodes.contains(gcluster + 1)) {
      nonlocal_nodes[gcluster + 1] += weight;
    } else {
      size.fetch_add(1, std::memory_order_relaxed);
      nonlocal_nodes[gcluster + 1] = weight;
    }
    //});
  }

  RECORD("nonlocal_nodes") StaticArray<GlobalNode> dense_nonlocal_nodes(size);
  std::size_t i = 0;
  for (const auto &[gcluster, weight] : nonlocal_nodes) {
    KASSERT(i < size);

    dense_nonlocal_nodes[i++] = {
        .gcluster = gcluster - 1,
        .weight = static_cast<NodeWeight>(weight),
    };
  }

  return dense_nonlocal_nodes;

  /*
  RECORD("node_position_buffer") StaticArray<NodeID> node_position_buffer(graph.total_n() + 1);

  graph.pfor_all_nodes([&](const NodeID lnode) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];

    if (!graph.is_owned_global_node(gcluster)) {
      node_position_buffer[lnode + 1] = 1;
    }
  });

  parallel::prefix_sum(
      node_position_buffer.begin(), node_position_buffer.end(), node_position_buffer.begin()
  );

  RECORD("nonlocal_nodes") StaticArray<GlobalNode> nonlocal_nodes(node_position_buffer.back());

  graph.pfor_all_nodes([&](const NodeID lnode) {
    const GlobalNodeID gcluster = lnode_to_gcluster[lnode];

    if (!graph.is_owned_global_node(gcluster)) {
      nonlocal_nodes[node_position_buffer[lnode]] = {
          .gcluster = gcluster,
          .weight = graph.is_owned_node(lnode) ? graph.node_weight(lnode) : 0,
      };
    }
  });

  return nonlocal_nodes; */
}

template <typename Graph>
StaticArray<GlobalEdge>
find_nonlocal_edges(const Graph &graph, const StaticArray<GlobalNodeID> &lnode_to_gcluster) {
  SCOPED_TIMER("Collect nonlocal edges");
  SCOPED_HEAP_PROFILER("Collect nonlocal edges");

  RECORD("edge_position_buffer") StaticArray<NodeID> edge_position_buffer(graph.n() + 1);

  graph.pfor_nodes([&](const NodeID lnode_u) {
    const GlobalNodeID gcluster_u = lnode_to_gcluster[lnode_u];

    NodeID nonlocal_neighbors_count = 0;
    if (!graph.is_owned_global_node(gcluster_u)) {
      graph.adjacent_nodes(lnode_u, [&](const NodeID lnode_v) {
        const GlobalNodeID gcluster_v = lnode_to_gcluster[lnode_v];

        if (gcluster_u != gcluster_v) {
          ++nonlocal_neighbors_count;
        }
      });
    }

    edge_position_buffer[lnode_u + 1] = nonlocal_neighbors_count;
  });

  parallel::prefix_sum(
      edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin()
  );

  RECORD("nonlocal_edges") StaticArray<GlobalEdge> nonlocal_edges(edge_position_buffer.back());

  graph.pfor_nodes([&](const NodeID lnode_u) {
    const GlobalNodeID gcluster_u = lnode_to_gcluster[lnode_u];

    if (!graph.is_owned_global_node(gcluster_u)) {
      NodeID pos = edge_position_buffer[lnode_u];

      graph.adjacent_nodes(lnode_u, [&](const NodeID lnode_v, const EdgeWeight w) {
        const GlobalNodeID gcluster_v = lnode_to_gcluster[lnode_v];

        if (gcluster_u != gcluster_v) {
          nonlocal_edges[pos++] = {
              .u = gcluster_u,
              .v = gcluster_v,
              .weight = w,
          };
        }
      });
    }
  });

  return nonlocal_edges;
}

void deduplicate_edge_list(StaticArray<GlobalEdge> &edges) {
  SCOPED_TIMER("Deduplicate edge list");
  SCOPED_HEAP_PROFILER("Deduplicate edge list");

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
  RECORD("edge_position_buffer") StaticArray<EdgeID> edge_position_buffer(edges.size());
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
  RECORD("tmp_nonlocal_edges")
  StaticArray<GlobalEdge> tmp_nonlocal_edges(edge_position_buffer.back() + 1);
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

void sort_node_list(StaticArray<GlobalNode> &nodes) {
  SCOPED_TIMER("Sort nodes");

  tbb::parallel_sort(nodes.begin(), nodes.end(), [&](const GlobalNode &lhs, const GlobalNode &rhs) {
    return lhs.gcluster < rhs.gcluster;
  });
}

template <typename Graph> void update_ghost_node_weights(Graph &graph) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Update ghost node weights");
  SCOPED_HEAP_PROFILER("Update ghost node weights");

  struct Message {
    NodeID local_node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID u) -> Message { return {u, graph.node_weight(u)}; },
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
  SCOPED_HEAP_PROFILER("Build node distribution");

  RECORD("distribution") StaticArray<T> distribution(mpi::get_comm_size(comm) + 1);

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

template <typename Graph>
StaticArray<NodeID> build_lcluster_to_lcnode_mapping(
    const Graph &graph,
    const StaticArray<GlobalNodeID> &lnode_to_gcluster,
    const StaticArray<GlobalNode> &local_nodes
) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Build lcluster_to_lcnode");
  SCOPED_HEAP_PROFILER("Build local cluster to local node mapping");

  RECORD("lcluster_to_lcnode") StaticArray<NodeID> lcluster_to_lcnode(graph.n());

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
          const GlobalNodeID gcluster = local_nodes[i].gcluster;
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

void localize_global_edge_list(StaticArray<GlobalEdge> &edges, const GlobalNodeID offset) {
  SCOPED_TIMER("Map edge list IDs to lcnode IDs");
  tbb::parallel_for<std::size_t>(0, edges.size(), [&](const std::size_t i) {
    edges[i].u -= offset;
  });
}

template <typename Graph>
std::pair<StaticArray<NodeID>, StaticArray<NodeID>> build_node_buckets(
    const Graph &graph,
    const StaticArray<NodeID> &lcluster_to_lcnode,
    const GlobalNodeID c_n,
    const StaticArray<GlobalEdge> &local_edges,
    const StaticArray<GlobalNodeID> &lnode_to_gcluster
) {
  SCOPED_TIMER("Bucket sort nodes by clusters");
  SCOPED_HEAP_PROFILER("Bucket sort nodes by clusters");

  RECORD("buckets_position_buffer") StaticArray<NodeID> buckets_position_buffer(c_n + 1);

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

  RECORD("buckets")
  StaticArray<NodeID> buckets(buckets_position_buffer.empty() ? 0 : buckets_position_buffer.back());

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
    const StaticArray<Element> &elements,
    MPI_Comm comm
) {
  TIMER_BARRIER(comm);

  SCOPED_TIMER("Exchange elements");
  SCOPED_HEAP_PROFILER("Exchange elements");

  const PEID size = mpi::get_comm_size(comm);

  std::vector<int> sendcounts(size);
  std::vector<int> sdispls(size);
  std::vector<int> recvcounts(size);
  std::vector<int> rdispls(size);

  std::copy(num_elements_for_pe.begin(), num_elements_for_pe.end(), sendcounts.begin());
  std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
  std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

  RECORD("recvbuf") StaticArray<Element> recvbuf(rdispls.back() + recvcounts.back());
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
      .rdispls = std::move(rdispls)
  };
}

template <typename Graph>
MigrationResult<GlobalNode>
migrate_nodes(const Graph &graph, const StaticArray<GlobalNode> &nonlocal_nodes) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Exchange nonlocal nodes");
  SCOPED_HEAP_PROFILER("Exchange nonlocal nodes");

  const PEID size = mpi::get_comm_size(graph.communicator());

  START_TIMER("Count messages");
  parallel::vector_ets<NodeID> num_nodes_for_pe_ets(size);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nonlocal_nodes.size()), [&](const auto &r) {
    auto &num_nodes_for_pe = num_nodes_for_pe_ets.local();

    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      const GlobalNodeID gcluster = nonlocal_nodes[i].gcluster;
      const PEID pe = graph.find_owner_of_global_node(gcluster);
      ++num_nodes_for_pe[pe];
    }
  });
  auto num_nodes_for_pe = num_nodes_for_pe_ets.combine(std::plus{});
  STOP_TIMER();

  return migrate_elements<GlobalNode>(num_nodes_for_pe, nonlocal_nodes, graph.communicator());
}

template <typename Graph>
MigrationResult<GlobalEdge> migrate_edges(
    const Graph &graph,
    const StaticArray<GlobalEdge> &nonlocal_edges,
    const StaticArray<GlobalNodeID> &c_node_distribution
) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Exchange nonlocal edges");
  SCOPED_HEAP_PROFILER("Exchange nonlocal edges");

  const PEID size = mpi::get_comm_size(graph.communicator());

  START_TIMER("Count messages");
  parallel::vector_ets<EdgeID> num_edges_for_pe_ets(size);
  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, nonlocal_edges.size()),
      [&](const auto &r) {
        auto &num_edges_for_pe = num_edges_for_pe_ets.local();
        PEID current_pe = 0;
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const GlobalNodeID u = nonlocal_edges[i].u;
          while (u >= c_node_distribution[current_pe + 1]) {
            ++current_pe;
          }
          ++num_edges_for_pe[current_pe];
        }
      },
      tbb::static_partitioner{}
  );
  auto num_edges_for_pe = num_edges_for_pe_ets.combine(std::plus{});
  STOP_TIMER();

  return migrate_elements<GlobalEdge>(num_edges_for_pe, nonlocal_edges, graph.communicator());
}

template <typename Graph>
MigratedNodesMapping exchange_migrated_nodes_mapping(
    const Graph &graph,
    const StaticArray<GlobalNode> &nonlocal_nodes,
    const MigrationResult<GlobalNode> &local_nodes,
    const StaticArray<NodeID> &lcluster_to_lcnode,
    const StaticArray<GlobalNodeID> &c_node_distribution
) {
  TIMER_BARRIER(graph.communicator());

  SCOPED_TIMER("Exchange node mapping for migrated nodes");
  SCOPED_HEAP_PROFILER("Exchange node mapping for migrated nodes");

  const PEID rank = mpi::get_comm_rank(graph.communicator());

  RECORD("their_nonlocal_to_gcnode")
  StaticArray<NodeMapping> their_nonlocal_to_gcnode(local_nodes.elements.size());

  RECORD("their_req_to_lcnode")
  StaticArray<NodeID> their_req_to_lcnode(their_nonlocal_to_gcnode.size());

  tbb::parallel_for<std::size_t>(0, local_nodes.elements.size(), [&](const std::size_t i) {
    const GlobalNodeID gcluster = local_nodes.elements[i].gcluster;
    const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
    const NodeID lcnode = lcluster_to_lcnode[lcluster];

    their_nonlocal_to_gcnode[i] = {
        .gcluster = gcluster,
        .gcnode = lcnode + c_node_distribution[rank],
    };
    their_req_to_lcnode[i] = lcnode;
  });

  RECORD("my_nonlocal_to_gcnode")
  StaticArray<NodeMapping> my_nonlocal_to_gcnode(nonlocal_nodes.size());

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
    const StaticArray<GlobalNodeID> &pe_overload,
    const StaticArray<GlobalNodeID> &pe_underload
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

  ScalableVector<PELoad> pe_load(size);
  StaticArray<GlobalNodeID> pe_overload(size + 1);
  StaticArray<GlobalNodeID> pe_underload(size + 1);

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

template <typename Graph>
void rebalance_cluster_placement(
    const Graph &graph,
    ClusterToCoarseMapper &cluster_mapper,
    const StaticArray<GlobalNodeID> &current_cnode_distribution,
    StaticArray<GlobalNodeID> &lnode_to_gcluster,
    const double max_cnode_imbalance,
    const double migrate_cnode_prefix
) {
  SCOPED_TIMER("Rebalance cluster assignment");
  SCOPED_HEAP_PROFILER("Rebalance cluster assignment");

  const auto shifts = compute_assignment_shifts(
      graph.node_distribution(), current_cnode_distribution, max_cnode_imbalance
  );

  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  graph.pfor_nodes_range([&](const auto &r) {
    auto &handle = cluster_mapper.handle();

    for (NodeID lnode = r.begin(); lnode != r.end(); ++lnode) {
      const GlobalNodeID old_gcluster = lnode_to_gcluster[lnode];

      GlobalNodeID old_gcnode = 0;
      PEID old_owner = 0;
      if (graph.is_owned_global_node(old_gcluster)) {
        old_gcnode = handle.local_gcluster_to_gcnode(old_gcluster);
        old_owner = rank;
      } else {
        old_gcnode = handle.nonlocal_gcluster_to_gcnode(old_gcluster);
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
      [&](const NodeID lnode) -> Message { return {lnode, lnode_to_gcluster[lnode]}; },
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

bool validate_clustering(
    const DistributedGraph &graph, const StaticArray<GlobalNodeID> &lnode_to_gcluster
) {
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

template <typename Graph>
std::unique_ptr<CoarseGraph> contract_clustering(
    const DistributedGraph &fine_graph,
    const Graph &graph,
    StaticArray<GlobalNodeID> &lnode_to_gcluster,
    const double max_cnode_imbalance = std::numeric_limits<double>::max(),
    const bool migrate_cnode_prefix = false,
    const bool force_perfect_cnode_balance = true
) {
  TIMER_BARRIER(graph.communicator());

  START_TIMER("Contract clustering");
  SCOPED_HEAP_PROFILER("Contract clustering");

  KASSERT(
      debug::validate_clustering(fine_graph, lnode_to_gcluster),
      "input clustering is invalid",
      assert::heavy
  );

  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  // Find all nodes (including ghost nodes) that belong to non-local clusters
  auto nonlocal_nodes = find_nonlocal_nodes(graph, lnode_to_gcluster);
  // .gcluster: the non-local gcluster ID
  // .weight: the weight of the node for local nodes, 0 for ghost nodes

  IF_STATS {
    const auto total_num_nonlocal_nodes =
        mpi::allreduce<GlobalNodeID>(nonlocal_nodes.size(), MPI_SUM, graph.communicator());
    STATS << "Total number of nonlocal nodes (including ghost nodes): " << total_num_nonlocal_nodes;
  }

  // We want to send these nodes to the cluster owning PEs. To do this efficiently, we must sort the
  // nodes first: this way, each PE should receive a contiguous range of our nodes
  sort_node_list(nonlocal_nodes);

  // Now do the actual exchange:
  auto migration_result_nodes = migrate_nodes(graph, nonlocal_nodes);
  auto &local_nodes = migration_result_nodes.elements;

  // After the exchange, each PE knows about all nodes that belong to its clusters, i.e., it knows
  // all of its non-empty clusters. Thus, we can now build the mapping from lcluster IDs to lcnode
  // IDs.
  RECORD("lcluster_to_lcnode")
  const StaticArray<NodeID> lcluster_to_lcnode =
      build_lcluster_to_lcnode_mapping(graph, lnode_to_gcluster, local_nodes);

  const NodeID c_n = lcluster_to_lcnode.empty() ? 0 : lcluster_to_lcnode.back() + 1;
  StaticArray<GlobalNodeID> c_node_distribution =
      build_distribution<GlobalNodeID>(c_n, graph.communicator());

  auto is_owned_gcnode = [&](const GlobalNodeID gcnode) {
    return gcnode >= c_node_distribution[rank] && gcnode < c_node_distribution[rank + 1];
  };

  // To construct the mapping from lnode IDs to lcnode IDs, we must acquire the mapping from
  // gcluster IDs to gcnode IDs for local nodes that belong to non-local clusters (i.e.,
  // nonlocal_nodes):
  auto mapping_exchange_result = exchange_migrated_nodes_mapping(
      graph, nonlocal_nodes, migration_result_nodes, lcluster_to_lcnode, c_node_distribution
  );
  auto &nonlocal_gcluster_to_gcnode_vec = mapping_exchange_result.my_nonlocal_to_gcnode;

  // Finally, we can construct the mapping for all relevant cluster IDs to coarse node IDs:
  START_TIMER("Construct cluster mapper");
  ClusterToCoarseMapper cluster_mapper(
      rank,
      graph.node_distribution(),
      c_node_distribution,
      lcluster_to_lcnode,
      nonlocal_gcluster_to_gcnode_vec
  );
  STOP_TIMER();

  // If the "natural" assignment of coarse nodes to PEs has too much imbalance,
  // we remap the cluster IDs to achieve perfect coarse node balance
  if (const double cnode_imbalance = compute_distribution_imbalance(c_node_distribution);
      cnode_imbalance > max_cnode_imbalance) {
    DBG << "Natural cluster assignment exceeds maximum coarse node imbalance: " << cnode_imbalance
        << " > " << max_cnode_imbalance << " --> rebalancing cluster assignment";

    const double max_imbalance = force_perfect_cnode_balance ? 1.0 : max_cnode_imbalance;

    rebalance_cluster_placement(
        graph,
        cluster_mapper,
        c_node_distribution,
        lnode_to_gcluster, // <<< inout
        max_imbalance,
        migrate_cnode_prefix
    );

    STOP_TIMER(); // Contract clustering timer

    // In some edge cases, rebalance_cluster_placement() might fail to balance the clusters to
    // max_imbalance (this is because the subgraph of a PE cannot grow in size during coarsening).
    // Thus, we accept any imbalance for the "rebalanced try" to avoid an infinite loop.
    return contract_clustering(fine_graph, graph, lnode_to_gcluster);
  }

  // Construct the mapping to coarse node IDs for our nodes (including ghost nodes)
  START_TIMER("Construct lnode_to_gcnode");
  RECORD("lnode_to_gcnode") StaticArray<GlobalNodeID> lnode_to_gcnode(graph.total_n());

  graph.pfor_all_nodes_range([&](const auto &r) {
    auto &handle = cluster_mapper.handle();

    for (NodeID lnode = r.begin(); lnode != r.end(); ++lnode) {
      const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
      lnode_to_gcnode[lnode] = handle.gcluster_to_gcnode(gcluster);
    }
  });
  STOP_TIMER();

  // At this point, we have all mappings that we need to construct the coarse graph; we can now
  // exchange edges:
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

  // find_nonlocal_edges() returns the edges with their gcluster IDs
  // Since we already have all the necessary mappings, we can remap these IDs to gcnode IDs
  START_TIMER("Map nonlocal edge IDs to gcnode IDs");
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nonlocal_edges.size()), [&](const auto &r) {
    auto &handle = cluster_mapper.handle();

    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      nonlocal_edges[i].u = handle.gcluster_to_gcnode(nonlocal_edges[i].u);
      nonlocal_edges[i].v = handle.gcluster_to_gcnode(nonlocal_edges[i].v);
    }
  });
  STOP_TIMER();

  auto migration_result_edges = migrate_edges(graph, nonlocal_edges, c_node_distribution);
  auto &local_edges = migration_result_edges.elements;

  // Sort again since we got edges from multiple PEs
  START_TIMER("Sort migrated local edges");
  tbb::parallel_sort(local_edges.begin(), local_edges.end(), [&](const auto &lhs, const auto &rhs) {
    return lhs.u < rhs.u;
  });
  STOP_TIMER();

  // Translate gcnode IDs (source vertex) to lcnode IDs (source vertex):
  localize_global_edge_list(local_edges, c_node_distribution[rank]);

  // Construct the ghost node mapping for the coarse graph:
  START_TIMER("Construct ghost node mapper");
  graph::GhostNodeMapper ghost_node_mapper(rank, c_node_distribution);

  tbb::parallel_invoke(
      [&] {
        graph.pfor_nodes([&](const NodeID lnode_u) {
          const GlobalNodeID gcnode_u = lnode_to_gcnode[lnode_u];
          if (!is_owned_gcnode(gcnode_u)) {
            return;
          }

          graph.adjacent_nodes(lnode_u, [&](const NodeID lnode_v) {
            const GlobalNodeID gcnode_v = lnode_to_gcnode[lnode_v];
            if (!is_owned_gcnode(gcnode_v)) {
              ghost_node_mapper.new_ghost_node(gcnode_v);
            }
          });
        });
      },
      [&] {
        tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
          const GlobalNodeID gcnode_v = local_edges[i].v;
          if (!is_owned_gcnode(gcnode_v)) {
            ghost_node_mapper.new_ghost_node(gcnode_v);
          }
        });
      }
  );

  auto ghost_node_mapper_result = ghost_node_mapper.finalize();
  STOP_TIMER();

  auto &c_global_to_ghost = ghost_node_mapper_result.global_to_ghost;
  auto &c_ghost_to_global = ghost_node_mapper_result.ghost_to_global;
  auto &c_ghost_owner = ghost_node_mapper_result.ghost_owner;
  const GlobalNodeID c_ghost_n = c_ghost_to_global.size();

  // From here on out, constructing the coarse graph is similar to the buffered shared-memory
  // implementation: first sort nodes by lcnode IDs
  auto bucket_sort_result =
      build_node_buckets(graph, lcluster_to_lcnode, c_n, local_edges, lnode_to_gcluster);
  auto &buckets_position_buffer = bucket_sort_result.first;
  auto &buckets = bucket_sort_result.second;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n + c_ghost_n);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector_ets([&] {
    return RatingMap<EdgeWeight, NodeID>(c_n + c_ghost_n);
  });

  struct LocalEdge {
    NodeID node;
    EdgeWeight weight;
  };

  NavigableLinkedList<NodeID, LocalEdge, ScalableVector> edge_buffer_ets;

  START_TIMER("Construct coarse edges");
  START_HEAP_PROFILER("Construct coarse edges");
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

          auto insert_edge = [&](const GlobalNodeID gcnode, const EdgeWeight weight) {
            if (is_owned_gcnode(gcnode)) {
              const auto lcnode = gcnode - c_node_distribution[rank];
              if (lcnode != lcu) {
                map[lcnode] += weight;
              }
            } else {
              auto lcnode_it = c_global_to_ghost.find(gcnode + 1);
              KASSERT(lcnode_it != c_global_to_ghost.end());
              const NodeID lcnode = (*lcnode_it).second;
              if (lcnode != lcu) {
                map[lcnode] += weight;
              }
            }
          };

          if (u < graph.n()) {
            c_u_weight += graph.node_weight(u);
            graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
              insert_edge(lnode_to_gcnode[v], w);
            });
          } else {
            for (std::size_t index = u - graph.n();
                 index < local_edges.size() && local_edges[index].u == lcu;
                 ++index) {
              insert_edge(local_edges[index].v, local_edges[index].weight);
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
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  START_TIMER("Integrate node weights of migrated nodes");
  tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
    const NodeID c_u = lcluster_to_lcnode[local_nodes[i].gcluster - graph.offset_n()];
    __atomic_fetch_add(&c_node_weights[c_u], local_nodes[i].weight, __ATOMIC_RELAXED);
  });
  STOP_TIMER();

  // Build edge distribution
  const EdgeID c_m = c_nodes.back();
  auto c_edge_distribution = build_distribution<GlobalEdgeID>(c_m, graph.communicator());
  DBG << "Coarse edge distribution: [" << c_edge_distribution << "]";

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse edges allocation");
  RECORD("c_edges") StaticArray<NodeID> c_edges(c_m);
  RECORD("c_edge_weights") StaticArray<EdgeWeight> c_edge_weights(c_m);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  // Finally, build coarse graph
  START_TIMER("Construct coarse graph");
  START_HEAP_PROFILER("Finalize coarse graph");
  auto all_buffered_nodes =                                                          //
      ts_navigable_list::combine<NodeID, LocalEdge, ScalableVector, ScalableVector>( //
          edge_buffer_ets                                                            //
      );                                                                             //

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

  DistributedCSRGraph coarse_csr_graph(
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
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  update_ghost_node_weights(coarse_csr_graph);

  STOP_TIMER(); // Contract clustering timer

  return std::make_unique<GlobalCoarseGraphImpl>(
      fine_graph,
      DistributedGraph(std::make_unique<DistributedCSRGraph>(std::move(coarse_csr_graph))),
      std::move(lnode_to_gcnode),
      MigratedNodes{
          .nodes = std::move(mapping_exchange_result.their_req_to_lcnode),
          .sendcounts = std::move(migration_result_nodes.recvcounts),
          .sdispls = std::move(migration_result_nodes.rdispls),
          .recvcounts = std::move(migration_result_nodes.sendcounts),
          .rdispls = std::move(migration_result_nodes.sdispls),
      }
  );
}

std::unique_ptr<CoarseGraph> contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    const CoarseningContext &c_ctx
) {
  return contract_clustering(
      graph,
      clustering,
      c_ctx.max_cnode_imbalance,
      c_ctx.migrate_cnode_prefix,
      c_ctx.force_perfect_cnode_balance
  );
}

std::unique_ptr<CoarseGraph> contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    double max_cnode_imbalance,
    bool migrate_cnode_prefix,
    bool force_perfect_cnode_balance
) {
  return graph.reified(
      [&](const DistributedCSRGraph &csr_graph) {
        return contract_clustering(
            graph,
            csr_graph,
            clustering,
            max_cnode_imbalance,
            migrate_cnode_prefix,
            force_perfect_cnode_balance
        );
      },
      [&](const DistributedCompressedGraph &compressed_graph) {
        return contract_clustering(
            graph,
            compressed_graph,
            clustering,
            max_cnode_imbalance,
            migrate_cnode_prefix,
            force_perfect_cnode_balance
        );
      }
  );
}

} // namespace kaminpar::dist
