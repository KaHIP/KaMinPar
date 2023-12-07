/*******************************************************************************
 * Allgather a distributed graph to each PE.
 *
 * @file:   allgather_graph.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#include "kaminpar-dist/graphutils/replicator.h"

#include <algorithm>

#include <mpi.h>

#include "kaminpar-mpi/utils.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/graphutils/synchronization.h"
#include "kaminpar-dist/metrics.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::dist {
SET_DEBUG(false);

std::unique_ptr<shm::Graph> allgather_graph(const DistributedGraph &graph) {
  return std::make_unique<shm::Graph>(replicate_graph_everywhere(graph));
}

std::pair<std::unique_ptr<shm::Graph>, std::unique_ptr<shm::PartitionedGraph>>
allgather_graph(const DistributedPartitionedGraph &p_graph) {
  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  auto shm_graph = allgather_graph(p_graph.graph());

  std::vector<int> counts(size);
  std::vector<int> displs(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    counts[pe] =
        asserting_cast<int>(p_graph.node_distribution(pe + 1) - p_graph.node_distribution(pe));
    displs[pe] = asserting_cast<int>(p_graph.node_distribution(pe));
  }
  displs.back() = asserting_cast<int>(p_graph.node_distribution(size));

  StaticArray<BlockID> shm_partition(displs.back());
  MPI_Allgatherv(
      p_graph.partition().data(),
      counts[rank],
      mpi::type::get<BlockID>(),
      shm_partition.data(),
      counts.data(),
      displs.data(),
      mpi::type::get<BlockID>(),
      p_graph.communicator()
  );

  auto shm_p_graph = std::make_unique<shm::PartitionedGraph>(
      *shm_graph.get(), p_graph.k(), std::move(shm_partition)
  );

  return {std::move(shm_graph), std::move(shm_p_graph)};
}

shm::Graph replicate_graph_everywhere(const DistributedGraph &graph) {
  KASSERT(
      graph.global_n() < std::numeric_limits<NodeID>::max(),
      "number of nodes exceeds int size",
      assert::always
  );
  KASSERT(
      graph.global_m() < std::numeric_limits<EdgeID>::max(),
      "number of edges exceeds int size",
      assert::always
  );
  MPI_Comm comm = graph.communicator();

  // copy edges array with global node IDs
  StaticArray<NodeID> remapped_edges(graph.m());
  graph.pfor_nodes([&](const NodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      remapped_edges[e] = graph.local_to_global_node(v);
    }
  });

  // gather graph
  StaticArray<shm::EdgeID> nodes(graph.global_n() + 1);
  StaticArray<shm::NodeID> edges(graph.global_m());

  const bool is_node_weighted =
      mpi::allreduce<std::uint8_t>(graph.is_node_weighted(), MPI_MAX, graph.communicator());
  const bool is_edge_weighted =
      mpi::allreduce<std::uint8_t>(graph.is_edge_weighted(), MPI_MAX, graph.communicator());

  StaticArray<shm::NodeWeight> node_weights(is_node_weighted * graph.global_n());
  StaticArray<shm::EdgeWeight> edge_weights(is_edge_weighted * graph.global_m());

  auto nodes_recvcounts = mpi::build_distribution_recvcounts(graph.node_distribution());
  auto nodes_displs = mpi::build_distribution_displs(graph.node_distribution());
  auto edges_recvcounts = mpi::build_distribution_recvcounts(graph.edge_distribution());
  auto edges_displs = mpi::build_distribution_displs(graph.edge_distribution());

  mpi::allgatherv(
      graph.raw_nodes().data(),
      asserting_cast<int>(graph.n()),
      nodes.data(),
      nodes_recvcounts.data(),
      nodes_displs.data(),
      comm
  );
  mpi::allgatherv(
      remapped_edges.data(),
      asserting_cast<int>(remapped_edges.size()),
      edges.data(),
      edges_recvcounts.data(),
      edges_displs.data(),
      comm
  );
  if (is_node_weighted) {
    KASSERT((graph.is_node_weighted() || graph.n() == 0));
    if constexpr (std::is_same_v<shm::NodeWeight, NodeWeight>) {
      mpi::allgatherv(
          graph.raw_node_weights().data(),
          asserting_cast<int>(graph.n()),
          node_weights.data(),
          nodes_recvcounts.data(),
          nodes_displs.data(),
          comm
      );
    } else {
      StaticArray<NodeWeight> node_weights_buffer(graph.global_n());
      mpi::allgatherv(
          graph.raw_node_weights().data(),
          asserting_cast<int>(graph.n()),
          node_weights_buffer.data(),
          nodes_recvcounts.data(),
          nodes_displs.data(),
          comm
      );
      tbb::parallel_for<std::size_t>(0, node_weights_buffer.size(), [&](const std::size_t i) {
        node_weights[i] = node_weights_buffer[i];
      });
    }
  }
  if (is_edge_weighted) {
    KASSERT((graph.is_edge_weighted() || graph.m() == 0));
    if constexpr (std::is_same_v<shm::EdgeWeight, EdgeWeight>) {
      mpi::allgatherv(
          graph.raw_edge_weights().data(),
          asserting_cast<int>(graph.m()),
          edge_weights.data(),
          edges_recvcounts.data(),
          edges_displs.data(),
          comm
      );
    } else {
      StaticArray<EdgeWeight> edge_weights_buffer(graph.global_m());
      mpi::allgatherv(
          graph.raw_edge_weights().data(),
          asserting_cast<int>(graph.m()),
          edge_weights_buffer.data(),
          edges_recvcounts.data(),
          edges_displs.data(),
          comm
      );
      tbb::parallel_for<std::size_t>(0, edge_weights_buffer.size(), [&](const std::size_t i) {
        edge_weights[i] = edge_weights_buffer[i];
      });
    }
  }
  nodes.back() = graph.global_m();

  // remap nodes array
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.global_n()), [&](const auto &r) {
    PEID pe = 0;
    for (NodeID u = r.begin(); u < r.end(); ++u) {
      while (u >= graph.node_distribution(pe + 1)) {
        KASSERT(pe < mpi::get_comm_size(comm));
        ++pe;
      }
      nodes[u] += graph.edge_distribution(pe);
    }
  });

  return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}

DistributedGraph replicate_graph(const DistributedGraph &graph, const int num_replications) {
  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  MPI_Comm new_comm = MPI_COMM_NULL;
  MPI_Comm_split(graph.communicator(), rank % num_replications, rank, &new_comm);
  const PEID new_size = mpi::get_comm_size(new_comm);
  const PEID new_rank = mpi::get_comm_rank(new_comm);

  // This communicator is used to mirror data between included PEs
  MPI_Comm primary_comm = MPI_COMM_NULL;
  MPI_Comm_split(graph.communicator(), new_rank, rank, &primary_comm);
  const PEID primary_size = mpi::get_comm_size(primary_comm);
  const PEID primary_rank = mpi::get_comm_rank(primary_comm);

  const bool need_secondary_comm = (size % num_replications != 0);
  const int is_secondary_participant =
      need_secondary_comm &&
      (rank + 1 == size || (new_size == size / num_replications && new_rank + 1 == new_size));
  MPI_Comm secondary_comm = MPI_COMM_NULL;
  NodeID secondary_num_nodes = 0;
  EdgeID secondary_num_edges = 0;
  PEID secondary_size = 0;
  PEID secondary_rank = 0;

  if (need_secondary_comm) {
    MPI_Comm_split(graph.communicator(), is_secondary_participant, rank, &secondary_comm);
    secondary_size = mpi::get_comm_size(secondary_comm);
    secondary_rank = mpi::get_comm_rank(secondary_comm);
  }
  const PEID secondary_root = secondary_size - 1;

  auto nodes_counts = mpi::build_counts_from_value<GlobalNodeID>(graph.n(), primary_comm);
  auto nodes_displs = mpi::build_displs_from_counts(nodes_counts);
  auto edges_counts = mpi::build_counts_from_value<GlobalEdgeID>(graph.m(), primary_comm);
  auto edges_displs = mpi::build_displs_from_counts(edges_counts);

  if (is_secondary_participant) {
    secondary_num_nodes = static_cast<NodeID>(nodes_displs.back());
    secondary_num_edges = static_cast<EdgeID>(edges_displs.back());

    MPI_Bcast(&secondary_num_nodes, 1, mpi::type::get<NodeID>(), secondary_root, secondary_comm);
    MPI_Bcast(&secondary_num_edges, 1, mpi::type::get<EdgeID>(), secondary_root, secondary_comm);

    if (secondary_rank == secondary_root) {
      secondary_num_nodes = 0;
      secondary_num_edges = 0;
    }
  }

  // Create edges array with global node IDs
  const GlobalEdgeID my_tmp_global_edges_offset = edges_displs[primary_rank];
  NoinitVector<GlobalNodeID> tmp_global_edges(edges_displs.back() + secondary_num_edges);
  graph.pfor_edges([&](const EdgeID e) {
    tmp_global_edges[my_tmp_global_edges_offset + e] =
        graph.local_to_global_node(graph.edge_target(e));
  });

  const bool is_node_weighted =
      mpi::allreduce<std::uint8_t>(graph.is_node_weighted(), MPI_MAX, graph.communicator());
  const bool is_edge_weighted =
      mpi::allreduce<std::uint8_t>(graph.is_edge_weighted(), MPI_MAX, graph.communicator());

  // Allocate memory for new graph
  StaticArray<EdgeID> nodes(nodes_displs.back() + secondary_num_nodes + 1);
  StaticArray<NodeID> edges(edges_displs.back() + secondary_num_edges);
  StaticArray<EdgeWeight> edge_weights;
  if (is_edge_weighted) {
    edge_weights.resize(edges.size());
  }

  // Exchange data -- except for node weights (need the number of ghost nodes
  // to allocate the vector)
  mpi::allgatherv(
      graph.raw_nodes().data(),
      asserting_cast<int>(graph.n()),
      nodes.data(),
      nodes_counts.data(),
      nodes_displs.data(),
      primary_comm
  );
  MPI_Allgatherv(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      tmp_global_edges.data(),
      edges_counts.data(),
      edges_displs.data(),
      mpi::type::get<GlobalNodeID>(),
      primary_comm
  );
  if (is_edge_weighted) {
    KASSERT(graph.is_edge_weighted() || graph.m() == 0);
    mpi::allgatherv(
        graph.raw_edge_weights().data(),
        asserting_cast<int>(graph.m()),
        edge_weights.data(),
        edges_counts.data(),
        edges_displs.data(),
        primary_comm
    );
  }

  // Set nodes guard
  nodes.back() = edges.size();

  // Offset received nodes arrays
  tbb::parallel_for<PEID>(0, primary_size, [&](const PEID p) {
    const NodeID offset = edges_displs[p];
    KASSERT(static_cast<std::size_t>(p + 1) < nodes_displs.size());

    tbb::parallel_for<NodeID>(nodes_displs[p], nodes_displs[p + 1], [&](const NodeID u) {
      KASSERT(u < nodes.size(), V(p) << V(nodes_displs) << V(nodes.size()));
      nodes[u] += offset;
    });
  });

  if (is_secondary_participant) {
    if (secondary_rank == secondary_root) {
      MPI_Bcast(
          nodes.data(),
          asserting_cast<int>(nodes.size() - 1),
          mpi::type::get<EdgeID>(),
          secondary_root,
          secondary_comm
      );
      MPI_Bcast(
          tmp_global_edges.data(),
          asserting_cast<int>(tmp_global_edges.size()),
          mpi::type::get<GlobalNodeID>(),
          secondary_root,
          secondary_comm
      );
      MPI_Bcast(
          edge_weights.data(),
          asserting_cast<int>(edge_weights.size()),
          mpi::type::get<EdgeWeight>(),
          secondary_root,
          secondary_comm
      );
    } else {
      MPI_Bcast(
          nodes.data() + nodes_displs.back(),
          asserting_cast<int>(secondary_num_nodes),
          mpi::type::get<EdgeID>(),
          secondary_root,
          secondary_comm
      );
      MPI_Bcast(
          tmp_global_edges.data() + edges_displs.back(),
          asserting_cast<int>(secondary_num_edges),
          mpi::type::get<NodeID>(),
          secondary_root,
          secondary_comm
      );
      MPI_Bcast(
          edge_weights.data() + edges_displs.back(),
          asserting_cast<int>(secondary_num_edges),
          mpi::type::get<EdgeWeight>(),
          secondary_root,
          secondary_comm
      );

      tbb::parallel_for<NodeID>(
          nodes_displs.back(),
          nodes_displs.back() + secondary_num_nodes,
          [&](const NodeID u) { nodes[u] += edges_displs.back(); }
      );
    }
  }

  // Create new node and edges distributions
  StaticArray<GlobalNodeID> node_distribution(new_size + 1);
  StaticArray<GlobalEdgeID> edge_distribution(new_size + 1);
  tbb::parallel_for<PEID>(0, new_size, [&](const PEID pe) { // no longer true
    const PEID of = std::min<PEID>(size, num_replications * (pe + 1));
    node_distribution[pe + 1] = graph.node_distribution(of);
    edge_distribution[pe + 1] = graph.edge_distribution(of);
  });
  node_distribution.back() = graph.node_distribution().back();
  edge_distribution.back() = graph.edge_distribution().back();

  DBG << "Node distribution: " << V(node_distribution);
  DBG << "Edge distribution: " << V(edge_distribution);

  // Remap edges to local nodes
  const GlobalEdgeID n0 = graph.node_distribution(rank) - nodes_displs[primary_rank];
  const GlobalEdgeID nf = n0 + nodes_displs.back() + secondary_num_nodes;
  graph::GhostNodeMapper ghost_node_mapper(new_rank, node_distribution);

  tbb::parallel_for<EdgeID>(0, tmp_global_edges.size(), [&](const EdgeID e) {
    const GlobalNodeID v = tmp_global_edges[e];
    if (v >= n0 && v < nf) {
      edges[e] = static_cast<NodeID>(v - n0);
    } else {
      DBG << "New edge to global node " << v;
      edges[e] = ghost_node_mapper.new_ghost_node(v);
    }
  });

  auto ghost_node_info = ghost_node_mapper.finalize();

  // Now that we know the number of ghost nodes: exchange node weights
  // The weights of ghost nodes are synchronized once the distributed graph data
  // structure was built
  const NodeID num_ghost_nodes = ghost_node_info.ghost_to_global.size();
  StaticArray<NodeWeight> node_weights(0);

  if (is_node_weighted) {
    KASSERT(graph.is_node_weighted() || graph.n() == 0);
    node_weights.resize(nodes_displs.back() + num_ghost_nodes);
    mpi::allgatherv(
        graph.raw_node_weights().data(),
        asserting_cast<int>(graph.n()),
        node_weights.data(),
        nodes_counts.data(),
        nodes_displs.data(),
        primary_comm
    );
  }

  DBG << V(ghost_node_info.ghost_owner) << V(ghost_node_info.ghost_to_global);
  for (const auto &[k, v] : ghost_node_info.global_to_ghost) {
    DBG << "Have mapping " << k << " --> " << v;
  }

  DistributedGraph new_graph(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      std::move(ghost_node_info.ghost_owner),
      std::move(ghost_node_info.ghost_to_global),
      std::move(ghost_node_info.global_to_ghost),
      false,
      new_comm
  );

  // Fix weights of ghost nodes
  if (is_node_weighted) {
    graph::synchronize_ghost_node_weights(new_graph);
  }

  KASSERT(debug::validate_graph(new_graph), "", assert::heavy);

  MPI_Comm_free(&primary_comm);
  if (need_secondary_comm) {
    MPI_Comm_free(&secondary_comm);
  }

  return new_graph;
}

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, DistributedPartitionedGraph p_graph) {
  // Create group with one PE of each replication
  const PEID group_size = mpi::get_comm_size(p_graph.communicator());
  const PEID group_rank = mpi::get_comm_rank(p_graph.communicator());
  const PEID size = mpi::get_comm_size(dist_graph.communicator());
  const PEID rank = mpi::get_comm_rank(dist_graph.communicator());
  const PEID num_replications = size / group_size;

  MPI_Comm inter_group_comm = MPI_COMM_NULL;
  MPI_Comm_split(dist_graph.communicator(), group_rank, rank, &inter_group_comm);
  const PEID inter_group_rank = mpi::get_comm_rank(inter_group_comm);

  // Find best partition
  const GlobalEdgeWeight my_cut = metrics::edge_cut(p_graph);
  struct ReductionMessage {
    long cut;
    int rank;
  };
  ReductionMessage best_cut_loc{my_cut, inter_group_rank};
  MPI_Allreduce(MPI_IN_PLACE, &best_cut_loc, 1, MPI_LONG_INT, MPI_MINLOC, inter_group_comm);

  // Compute partition distribution for p_graph --> dist_graph
  NoinitVector<int> send_counts(num_replications);
  for (PEID pe = group_rank * num_replications; pe < (group_rank + 1) * num_replications; ++pe) {
    const PEID first_pe = group_rank * num_replications;
    send_counts[pe - first_pe] = asserting_cast<int>(
        dist_graph.node_distribution(pe + 1) - dist_graph.node_distribution(pe)
    );
  }
  NoinitVector<int> send_displs = mpi::build_displs_from_counts(send_counts);
  int recv_count = asserting_cast<int>(dist_graph.n());

  // Scatter best partition
  auto partition = p_graph.take_partition();
  StaticArray<BlockID> new_partition(dist_graph.total_n());
  MPI_Scatterv(
      partition.data(),
      send_counts.data(),
      send_displs.data(),
      mpi::type::get<BlockID>(),
      new_partition.data(),
      recv_count,
      mpi::type::get<BlockID>(),
      best_cut_loc.rank,
      inter_group_comm
  );

  // Create partitioned dist_graph
  DistributedPartitionedGraph p_dist_graph(&dist_graph, p_graph.k(), std::move(new_partition));

  // Synchronize ghost node assignment
  graph::synchronize_ghost_node_block_ids(p_dist_graph);

  MPI_Comm_free(&inter_group_comm);
  return p_dist_graph;
}

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, shm::PartitionedGraph shm_p_graph) {
  KASSERT(
      dist_graph.global_n() < static_cast<GlobalNodeID>(std::numeric_limits<NodeID>::max()),
      "partition size exceeds int size",
      assert::always
  );
  MPI_Comm comm = dist_graph.communicator();

  const PEID rank = mpi::get_comm_rank(comm);
  const auto shm_cut = asserting_cast<EdgeWeight>(shm::metrics::edge_cut(shm_p_graph));

  // Find PE with best partition
  struct ReductionMessage {
    long cut;
    int rank;
  };

  ReductionMessage local = {.cut = shm_cut, .rank = rank};
  ReductionMessage global = {.cut = 0, .rank = 0};

  MPI_Allreduce(&local, &global, 1, MPI_LONG_INT, MPI_MINLOC, comm);

  // Broadcast best partition
  auto partition = shm_p_graph.take_raw_partition();
  MPI_Bcast(
      partition.data(),
      static_cast<int>(dist_graph.global_n()),
      mpi::type::get<shm::BlockID>(),
      global.rank,
      comm
  );

  // Create distributed partition
  StaticArray<BlockID> dist_partition(dist_graph.total_n());
  dist_graph.pfor_nodes(0, dist_graph.total_n(), [&](const NodeID u) {
    dist_partition[u] = partition[dist_graph.local_to_global_node(u)];
  });

  // Create distributed partitioned graph
  return {&dist_graph, shm_p_graph.k(), std::move(dist_partition)};
}

DistributedPartitionedGraph distribute_partition(
    const DistributedGraph &graph,
    const BlockID k,
    const StaticArray<shm::BlockID> &global_partition,
    const PEID root
) {
  const PEID rank = mpi::get_comm_rank(graph.communicator());
  const PEID size = mpi::get_comm_size(graph.communicator());

  // Compute partition distribution for p_graph --> dist_graph
  std::vector<int> scounts(size);
  std::vector<int> sdispls(size);
  for (PEID pe = 0; pe < size; ++pe) {
    scounts[pe] =
        asserting_cast<int>(graph.node_distribution(pe + 1) - graph.node_distribution(pe));
  }
  std::exclusive_scan(scounts.begin(), scounts.end(), sdispls.begin(), 0);
  const int rcount = asserting_cast<int>(graph.n());

  StaticArray<BlockID> local_partition(graph.total_n());

  MPI_Scatterv(
      (rank == root ? global_partition.data() : nullptr),
      scounts.data(),
      sdispls.data(),
      mpi::type::get<BlockID>(),
      local_partition.data(),
      rcount,
      mpi::type::get<BlockID>(),
      root,
      graph.communicator()
  );

  // Create partitioned dist_graph
  DistributedPartitionedGraph p_graph(&graph, k, std::move(local_partition));
  graph::synchronize_ghost_node_block_ids(p_graph);
  return p_graph;
}
} // namespace kaminpar::dist
