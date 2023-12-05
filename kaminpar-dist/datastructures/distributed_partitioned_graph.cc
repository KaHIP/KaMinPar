/*******************************************************************************
 * Wrapper with a dynamic partition for a static distributed graph.
 *
 * @file:   distributed_partitioned_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.06.2023
 ******************************************************************************/
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

#include <tbb/parallel_for.h>

#include "kaminpar-dist/logger.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/parallel/vector_ets.h"

namespace kaminpar::dist {
void DistributedPartitionedGraph::init_block_weights() {
  parallel::vector_ets<BlockWeight> local_block_weights_ets(k());

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const auto &r) {
    auto &local_block_weights = local_block_weights_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      local_block_weights[block(u)] += node_weight(u);
    }
  });
  auto local_block_weights = local_block_weights_ets.combine(std::plus{});

  scalable_vector<BlockWeight> global_block_weights_nonatomic(k());
  mpi::allreduce(
      local_block_weights.data(),
      global_block_weights_nonatomic.data(),
      k(),
      MPI_SUM,
      communicator()
  );

  _block_weights.resize(k());
  pfor_blocks([&](const BlockID b) { _block_weights[b] = global_block_weights_nonatomic[b]; });
}

namespace debug {
bool validate_partition(const DistributedPartitionedGraph &p_graph) {
  MPI_Comm comm = p_graph.communicator();
  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  {
    const BlockID root_k = mpi::bcast(p_graph.k(), 0, comm);
    if (root_k != p_graph.k()) {
      LOG_ERROR << "on PE " << rank << ": number of blocks (" << p_graph.k()
                << ") differs from number of blocks on the root PE (" << root_k << ")";
      return false;
    }
  }

  mpi::barrier(comm);

  {
    for (const NodeID u : p_graph.all_nodes()) {
      if (p_graph.block(u) >= p_graph.k()) {
        LOG_ERROR << "on PE " << rank << ": node " << u << " assigned to invalid block "
                  << p_graph.block(u);
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
    StaticArray<BlockWeight> recomputed_block_weights(p_graph.k());
    for (const NodeID u : p_graph.nodes()) {
      recomputed_block_weights[p_graph.block(u)] += p_graph.node_weight(u);
    }
    MPI_Allreduce(
        MPI_IN_PLACE,
        recomputed_block_weights.data(),
        asserting_cast<int>(p_graph.k()),
        mpi::type::get<BlockWeight>(),
        MPI_SUM,
        comm
    );

    for (const BlockID b : p_graph.blocks()) {
      if (p_graph.block_weight(b) != recomputed_block_weights[b]) {
        LOG_ERROR << "on PE " << rank << ": expected weight of block " << b << " is "
                  << p_graph.block_weight(b) << ", but actual weight is "
                  << recomputed_block_weights[b];
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
    // Build a global partition array on the root PE
    StaticArray<BlockID> global_partition(0);
    if (rank == 0) {
      global_partition.resize(p_graph.global_n());
    }

    std::vector<int> displs(p_graph.node_distribution().begin(), p_graph.node_distribution().end());
    std::vector<int> counts(size);
    for (PEID pe = 0; pe < size; ++pe) {
      counts[pe] = displs[pe + 1] - displs[pe];
    }

    MPI_Gatherv(
        p_graph.partition().data(),
        asserting_cast<int>(p_graph.n()),
        mpi::type::get<BlockID>(),
        global_partition.data(),
        counts.data(),
        displs.data(),
        mpi::type::get<BlockID>(),
        0,
        comm
    );

    // Collect ghost node assignments on the root PE
    struct NodeAssignment {
      GlobalNodeID node;
      BlockID block;
    };
    std::vector<NodeAssignment> sendbuf;
    for (const NodeID ghost_u : p_graph.ghost_nodes()) {
      const GlobalNodeID global = p_graph.local_to_global_node(ghost_u);
      const BlockID block = p_graph.block(ghost_u);

      if (rank == 0) {
        if (global_partition[global] != block) {
          KASSERT(global < global_partition.size());
          LOG_ERROR << "inconsistent assignment of node " << global
                    << " on PE 0: local assignment to block " << block
                    << " inconsistent with actual assignment to block " << global_partition[global];
          return false;
        }
      } else {
        sendbuf.push_back({global, block});
      }
    }

    if (rank == 0) {
      for (int pe = 1; pe < size; ++pe) {
        const auto recvbuf = mpi::probe_recv<NodeAssignment>(pe, 0, comm);

        for (const auto &[global, block] : recvbuf) {
          if (global_partition[global] != block) {
            LOG_ERROR << "inconsistent assignment of node " << global
                      << " on PE 0: local assignment to block " << block
                      << " inconsistent with actual assignment to block "
                      << global_partition[global];
            return false;
          }
        }
      }
    } else {
      mpi::send(sendbuf.data(), sendbuf.size(), 0, 0, comm);
    }
  }

  mpi::barrier(comm);
  return true;
}
} // namespace debug
} // namespace kaminpar::dist
