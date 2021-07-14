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

bool validate(const DistributedGraph &graph, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);

  // check global n, global m
  {
    const auto recv_global_n = mpi::gather(graph.global_n());
    ALWAYS_ASSERT_ROOT(all_equal(recv_global_n));

    const auto recv_global_m = mpi::gather(graph.global_m());
    ALWAYS_ASSERT_ROOT(all_equal(recv_global_m));
  }

  return true;
}

bool validate_partition(const DistributedPartitionedGraph &p_graph, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);

  // check whether each PE knows the same block count
  {
    const auto recv_k = mpi::gather(p_graph.k());
    ALWAYS_ASSERT_ROOT(all_equal(recv_k));
  }

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

  // check whether assignment of ghost nodes is consistent
  {
    // collect partition on root
    scalable_vector<DBlockID> recv_partition;
    if (ROOT(rank)) { recv_partition.resize(p_graph.global_n()); }
    const scalable_vector<DBlockID> send_partition = p_graph.partition_copy();
    mpi::gather(send_partition.data(), p_graph.n(), recv_partition.data(), p_graph.n(), 0, comm);

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
        int count = mpi::get_count<std::uint64_t>(&status);
        scalable_vector<std::uint64_t> recv_buffer(count);
        mpi::recv(recv_buffer.data(), count, pe, 0);

        // now validate received data
        for (int i = 0; i < count; i += 2) {
          const DNodeID global_u = static_cast<DNodeID>(recv_buffer[i]);
          const DBlockID b = static_cast<DBlockID>(recv_buffer[i + 1]);
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

  MPI_Barrier(comm);
  return true;
}
} // namespace dkaminpar::graph::debug