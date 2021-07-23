/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "dkaminpar/datastructure/distributed_graph.h"

#include "dkaminpar/mpi_utils.h"
#include "dkaminpar/mpi_wrapper.h"

#include <ranges>

namespace dkaminpar::graph::debug {
SET_DEBUG(true);

namespace {
template<std::ranges::range R>
bool all_equal(const R &r) {
  return std::ranges::adjacent_find(r, std::not_equal_to{}) == std::ranges::end(r);
}
} // namespace

bool validate(const DistributedGraph &graph, const int root, MPI_Comm comm) {
  mpi::barrier(comm);

  const auto [size, rank] = mpi::get_comm_info(comm);

  // check global n, global m
  ALWAYS_ASSERT(mpi::bcast(graph.global_n(), root, comm) == graph.global_n());
  ALWAYS_ASSERT(mpi::bcast(graph.global_m(), root, comm) == graph.global_m());

  // check global node distribution
  ALWAYS_ASSERT(static_cast<int>(graph.node_distribution().size()) == size + 1);
  ALWAYS_ASSERT(graph.node_distribution().front() == 0);
  ALWAYS_ASSERT(graph.node_distribution().back() == graph.global_n());
  for (PEID pe = 1; pe < size + 1; ++pe) {
    ALWAYS_ASSERT(mpi::bcast(graph.node_distribution(pe), root, comm) == graph.node_distribution(pe));
    ALWAYS_ASSERT(rank + 1 != pe || graph.node_distribution(pe) - graph.node_distribution(pe - 1) == graph.n());
  }

  // check global edge distribution
  ALWAYS_ASSERT(static_cast<int>(graph.edge_distribution().size()) == size + 1);
  ALWAYS_ASSERT(graph.edge_distribution().front() == 0);
  ALWAYS_ASSERT(graph.edge_distribution().back() == graph.global_m());
  for (PEID pe = 1; pe < size + 1; ++pe) {
    ALWAYS_ASSERT(mpi::bcast(graph.edge_distribution(pe), root, comm) == graph.edge_distribution(pe));
    ALWAYS_ASSERT(rank + 1 != pe || graph.edge_distribution(pe) - graph.edge_distribution(pe - 1) == graph.m());
  }

  // check that ghost nodes are actually ghost nodes
  for (DNodeID ghost_u : graph.ghost_nodes()) { ALWAYS_ASSERT(graph.ghost_owner(ghost_u) != rank); }

  // check node weight of ghost nodes
  {
    struct GhostNodeWeightMessage {
      DNodeID u;
      DNodeWeight weight;
    };


    std::vector<std::vector<GhostNodeWeightMessage>> send_buffers(size);
    for (DNodeID ghost_u : graph.ghost_nodes()) {
      const PEID owner = graph.ghost_owner(ghost_u);
      send_buffers[owner].emplace_back(graph.global_node(ghost_u), graph.node_weight(ghost_u));
    }
    ALWAYS_ASSERT(send_buffers[rank].empty());

    mpi::exchange<scalable_vector>(send_buffers, 0, [&](const PEID, const auto &data) {
      for (const auto [global_u, weight] : data) {
        const DNodeID local_u = graph.local_node(global_u);
        ALWAYS_ASSERT(graph.offset_n() <= global_u && global_u < graph.offset_n() + graph.n());
        ALWAYS_ASSERT(graph.node_weight(local_u) == weight);
      }
    });
  }

  mpi::barrier(comm);
  return true;
}

bool validate_partition(const DistributedPartitionedGraph &p_graph, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);

  // check whether each PE knows the same block count
  {
    const auto recv_k = mpi::gather(p_graph.k());
    ALWAYS_ASSERT_ROOT(all_equal(recv_k));
  }

  mpi::barrier(comm);
  LOG << "Every PE knows the same number of blocks";

  // check whether each PE has the same block weights
  {
    scalable_vector<DBlockWeight> recv_block_weights;
    if (ROOT(rank)) { recv_block_weights.resize(size * p_graph.k()); }
    const scalable_vector<DBlockWeight> send_block_weights = p_graph.block_weights_copy();
    mpi::gather(send_block_weights.data(), p_graph.k(), recv_block_weights.data(), p_graph.k(), 0, comm);

    if (ROOT(rank)) {
      DBG << "block_weights=" << recv_block_weights;
      for (const DBlockID b : p_graph.blocks()) {
        for (int pe = 0; pe < size; ++pe) {
          const DBlockWeight expected = recv_block_weights[b];
          const DBlockWeight actual = recv_block_weights[p_graph.k() * pe + b];
          ALWAYS_ASSERT(expected == actual) << "for PE " << pe << ", block " << b << ": expected weight " << expected
                                            << " (weight on root), got weight " << actual;
        }
      }
    }
  }

  mpi::barrier(comm);
  LOG << "Every PE knows the same block weights";

  // check whether block weights are actually correct
  {
    scalable_vector<DBlockWeight> send_block_weights(p_graph.k());
    for (const DNodeID u : p_graph.nodes()) { send_block_weights[p_graph.block(u)] += p_graph.node_weight(u); }
    DBG << V(send_block_weights) << V(p_graph.block_weights_copy());
    scalable_vector<DBlockWeight> recv_block_weights;
    if (ROOT(rank)) { recv_block_weights.resize(p_graph.k()); }
    mpi::reduce(send_block_weights.data(), recv_block_weights.data(), p_graph.k(), MPI_SUM, 0, comm);
    if (ROOT(rank)) {
      for (const DBlockID b : p_graph.blocks()) {
        ALWAYS_ASSERT(p_graph.block_weight(b) == recv_block_weights[b])
            << V(b) << V(p_graph.block_weight(b)) << V(recv_block_weights[b]);
      }
    }
  }

  mpi::barrier(comm);
  LOG << "Every PE knows the right block weights";

  // check whether assignment of ghost nodes is consistent
  {
    // collect partition on root
    scalable_vector<DBlockID> recv_partition;
    if (ROOT(rank)) { recv_partition.resize(p_graph.global_n()); }

    const auto recvcounts = mpi::build_distribution_recvcounts(p_graph.node_distribution());
    const auto displs = mpi::build_distribution_displs(p_graph.node_distribution());
    MPI_Gatherv(p_graph.partition().data(), p_graph.n(), MPI_UINT32_T, recv_partition.data(), recvcounts.data(),
                displs.data(), MPI_UINT32_T, 0, comm);

    // next, each PE validates the block of its ghost nodes by sending them to root
    scalable_vector<std::uint64_t> send_buffer;
    send_buffer.reserve(p_graph.ghost_n() * 2);
    for (const DNodeID ghost_u : p_graph.ghost_nodes()) {
      if (ROOT(rank)) { // root can validate locally
        ALWAYS_ASSERT(p_graph.block(ghost_u) == recv_partition[p_graph.global_node(ghost_u)]);
      } else {
        send_buffer.push_back(p_graph.global_node(ghost_u));
        send_buffer.push_back(p_graph.block(ghost_u));
      }
    }

    // exchange messages and validate
    if (ROOT(rank)) {
      for (int pe = 1; pe < size; ++pe) { // recv from all but root
        MPI_Status status = mpi::probe(pe, 0);
        int count = mpi::get_count<std::uint64_t>(status);
        scalable_vector<std::uint64_t> recv_buffer(count);
        mpi::recv(recv_buffer.data(), count, pe, 0);

        // now validate received data
        for (int i = 0; i < count; i += 2) {
          const auto global_u = static_cast<DNodeID>(recv_buffer[i]);
          const auto b = static_cast<DBlockID>(recv_buffer[i + 1]);
          ALWAYS_ASSERT(recv_partition[global_u] == b)
              << "on PE " << pe << ": ghost node " << global_u << " is placed in block " << b
              << ", but on its owner PE, it is placed in block " << recv_partition[global_u];
        }
        DBG << "Validated " << count / 2 << " ghost nodes from PE " << pe;
      }
    } else {
      mpi::send(send_buffer.data(), send_buffer.size(), 0, 0);
    }
  }

  mpi::barrier(comm);
  LOG << "Every PE has its ghost nodes in the right block";

  return true;
}
} // namespace dkaminpar::graph::debug