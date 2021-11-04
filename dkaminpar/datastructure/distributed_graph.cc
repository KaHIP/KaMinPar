/*******************************************************************************
 * @file:   distributed_graph.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Static distributed graph data structure.
 ******************************************************************************/
#include "dkaminpar/datastructure/distributed_graph.h"

#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"

#include <iomanip>
#include <ranges>

namespace dkaminpar {
void DistributedGraph::print() const {
  std::ostringstream buf;

  const int w = std::ceil(std::log10(global_n()));

  buf << "n=" << n() << " m=" << m() << " ghost_n=" << ghost_n() << " total_n=" << total_n() << "\n";
  buf << "--------------------------------------------------------------------------------\n";
  for (const NodeID u : all_nodes()) {
    const char u_prefix = is_owned_node(u) ? ' ' : '!';
    buf << u_prefix << "L" << std::setw(w) << u << " G" << std::setw(w) << local_to_global_node(u) << " W"
        << std::setw(w) << node_weight(u);

    if (is_owned_node(u)) {
      buf << " | ";
      for (const auto [e, v] : neighbors(u)) {
        const char v_prefix = is_owned_node(v) ? ' ' : '!';
        buf << v_prefix << "L" << std::setw(w) << v << " G" << std::setw(w) << local_to_global_node(v) << " EW"
            << std::setw(w) << edge_weight(e) << "\t";
      }
      if (degree(u) == 0) {
        buf << "<empty>";
      }
    }
    buf << "\n";
  }
  buf << "--------------------------------------------------------------------------------\n";
  SLOG << buf.str();
}
} // namespace dkaminpar

namespace dkaminpar::graph::debug {
SET_DEBUG(false);

namespace {
template <std::ranges::range R> bool all_equal(const R &r) {
  return std::ranges::adjacent_find(r, std::not_equal_to{}) == std::ranges::end(r);
}
} // namespace

bool validate(const DistributedGraph &graph, const int root) {
  MPI_Comm comm = graph.communicator();
  mpi::barrier(comm);

  const auto [size, rank] = mpi::get_comm_info(comm);

  // check global n, global m
  DBG << "Checking global n, m";
  ALWAYS_ASSERT(mpi::bcast(graph.global_n(), root, comm) == graph.global_n());
  ALWAYS_ASSERT(mpi::bcast(graph.global_m(), root, comm) == graph.global_m());

  // check global node distribution
  DBG << "Checking node distribution";
  ALWAYS_ASSERT(static_cast<int>(graph.node_distribution().size()) == size + 1);
  ALWAYS_ASSERT(graph.node_distribution().front() == 0);
  ALWAYS_ASSERT(graph.node_distribution().back() == graph.global_n());
  for (PEID pe = 1; pe < size + 1; ++pe) {
    ALWAYS_ASSERT(mpi::bcast(graph.node_distribution(pe), root, comm) == graph.node_distribution(pe));
    ALWAYS_ASSERT(rank + 1 != pe || graph.node_distribution(pe) - graph.node_distribution(pe - 1) == graph.n());
  }

  // check global edge distribution
  DBG << "Checking edge distribution";
  ALWAYS_ASSERT(static_cast<int>(graph.edge_distribution().size()) == size + 1);
  ALWAYS_ASSERT(graph.edge_distribution().front() == 0);
  ALWAYS_ASSERT(graph.edge_distribution().back() == graph.global_m());
  for (PEID pe = 1; pe < size + 1; ++pe) {
    ALWAYS_ASSERT(mpi::bcast(graph.edge_distribution(pe), root, comm) == graph.edge_distribution(pe));
    ALWAYS_ASSERT(rank + 1 != pe || graph.edge_distribution(pe) - graph.edge_distribution(pe - 1) == graph.m());
  }

  // check that ghost nodes are actually ghost nodes
  DBG << "Checking ghost nodes";
  for (NodeID ghost_u : graph.ghost_nodes()) {
    ALWAYS_ASSERT(graph.ghost_owner(ghost_u) != rank);
  }

  // check node weight of ghost nodes
  DBG << "Checking node weights of ghost nodes";
  {
    struct GhostNodeWeightMessage {
      GlobalNodeID global_u;
      NodeWeight weight;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeWeightMessage>(
        graph,
        [&](const NodeID u) -> GhostNodeWeightMessage {
          return {.global_u = graph.local_to_global_node(u), .weight = graph.node_weight(u)};
        },
        [&](const auto buffer) {
          for (const auto [global_u, weight] : buffer) {
            ALWAYS_ASSERT(graph.contains_global_node(global_u));
            const NodeID local_u = graph.global_to_local_node(global_u);
            ALWAYS_ASSERT(graph.node_weight(local_u) == weight);
          }
        });
  }

  // check that edges to ghost nodes exist in both directions
  DBG << "Checking edges to ghost nodes";
  {
    struct GhostNodeEdge {
      GlobalNodeID owned_node;
      GlobalNodeID ghost_node;
    };

    mpi::graph::sparse_alltoall_interface_to_ghost<GhostNodeEdge>(
        graph,
        [&](const NodeID u, const EdgeID, const NodeID v) -> GhostNodeEdge {
          return {.owned_node = graph.local_to_global_node(u), .ghost_node = graph.local_to_global_node(v)};
        },
        [&](const auto recv_buffer, const PEID pe) {
          for (const auto [ghost_node, owned_node] : recv_buffer) { // NOLINT: roles are swapped on receiving PE
            ALWAYS_ASSERT(graph.contains_global_node(ghost_node));
            ALWAYS_ASSERT(graph.contains_global_node(owned_node));

            const NodeID local_owned_node = graph.global_to_local_node(owned_node);
            const NodeID local_ghost_node = graph.global_to_local_node(ghost_node);

            bool found = false;
            for (const auto v : graph.adjacent_nodes(local_owned_node)) {
              if (v == local_ghost_node) {
                found = true;
                break;
              }
            }
            ALWAYS_ASSERT(found) << "Node " << local_owned_node << " (g " << owned_node << ") "
                                 << "is expected to be adjacent to " << local_ghost_node << " (g " << ghost_node << ") "
                                 << "due to an edge on PE " << pe << ", but is not";
          }
        });
  }

  mpi::barrier(comm);
  return true;
}

bool validate_partition(const DistributedPartitionedGraph &p_graph) {
  MPI_Comm comm = p_graph.communicator();
  const auto [size, rank] = mpi::get_comm_info(comm);

  {
    DBG << "Check that each PE knows the same block count";
    const auto recv_k = mpi::gather(p_graph.k());
    ALWAYS_ASSERT_ROOT(all_equal(recv_k));
    mpi::barrier(comm);
  }

  {
    DBG << "Check that block IDs are OK";
    for (const NodeID u : p_graph.all_nodes()) {
      ALWAYS_ASSERT(p_graph.block(u) < p_graph.k());
    }
  }

  {
    DBG << "Check that each PE has the same block weights";

    scalable_vector<BlockWeight> recv_block_weights;
    if (ROOT(rank)) {
      recv_block_weights.resize(size * p_graph.k());
    }
    const scalable_vector<BlockWeight> send_block_weights = p_graph.block_weights_copy();
    mpi::gather(send_block_weights.data(), static_cast<int>(p_graph.k()), recv_block_weights.data(),
                static_cast<int>(p_graph.k()), 0, comm);

    if (ROOT(rank)) {
      for (const BlockID b : p_graph.blocks()) {
        for (int pe = 0; pe < size; ++pe) {
          const BlockWeight expected = recv_block_weights[b];
          const BlockWeight actual = recv_block_weights[p_graph.k() * pe + b];
          ALWAYS_ASSERT(expected == actual) << "for PE " << pe << ", block " << b << ": expected weight " << expected
                                            << " (weight on root), got weight " << actual;
        }
      }
    }

    mpi::barrier(comm);
  }

  {
    DBG << "Check that block weights are actually correct";

    scalable_vector<BlockWeight> send_block_weights(p_graph.k());
    for (const NodeID u : p_graph.nodes()) {
      send_block_weights[p_graph.block(u)] += p_graph.node_weight(u);
    }
    scalable_vector<BlockWeight> recv_block_weights;
    if (ROOT(rank)) {
      recv_block_weights.resize(p_graph.k());
    }
    mpi::reduce(send_block_weights.data(), recv_block_weights.data(), static_cast<int>(p_graph.k()), MPI_SUM, 0, comm);
    if (ROOT(rank)) {
      for (const BlockID b : p_graph.blocks()) {
        ALWAYS_ASSERT(p_graph.block_weight(b) == recv_block_weights[b])
            << V(b) << V(p_graph.block_weight(b)) << V(recv_block_weights[b]);
      }
    }

    mpi::barrier(comm);
  }

  {
    DBG << "Check whether the assignment of ghost nodes is consistent";

    // collect partition on root
    scalable_vector<BlockID> recv_partition;
    if (ROOT(rank)) {
      recv_partition.resize(p_graph.global_n());
    }

    const auto recvcounts = mpi::build_distribution_recvcounts(p_graph.node_distribution());
    const auto displs = mpi::build_distribution_displs(p_graph.node_distribution());
    mpi::gatherv(p_graph.partition().data(), static_cast<int>(p_graph.n()), recv_partition.data(), recvcounts.data(),
                 displs.data(), 0, comm);

    // next, each PE validates the block of its ghost nodes by sending them to root
    scalable_vector<std::uint64_t> send_buffer;
    send_buffer.reserve(p_graph.ghost_n() * 2);
    for (const NodeID ghost_u : p_graph.ghost_nodes()) {
      if (ROOT(rank)) { // root can validate locally
        ALWAYS_ASSERT(p_graph.block(ghost_u) == recv_partition[p_graph.local_to_global_node(ghost_u)]);
      } else {
        send_buffer.push_back(p_graph.local_to_global_node(ghost_u));
        send_buffer.push_back(p_graph.block(ghost_u));
      }
    }

    // exchange messages and validate
    if (ROOT(rank)) {
      for (int pe = 1; pe < size; ++pe) { // recv from all but root
        const auto recv_buffer = mpi::probe_recv<std::uint64_t>(pe, 0, MPI_STATUS_IGNORE, comm);

        // now validate received data
        for (std::size_t i = 0; i < recv_buffer.size(); i += 2) {
          const auto global_u = static_cast<GlobalNodeID>(recv_buffer[i]);
          const auto b = static_cast<BlockID>(recv_buffer[i + 1]);
          ALWAYS_ASSERT(recv_partition[global_u] == b)
              << "on PE " << pe << ": ghost node " << global_u << " is placed in block " << b
              << ", but on its owner PE, it is placed in block " << recv_partition[global_u];
        }
      }
    } else {
      mpi::send(send_buffer.data(), send_buffer.size(), 0, 0);
    }

    mpi::barrier(comm);
  }

  return true;
}
} // namespace dkaminpar::graph::debug