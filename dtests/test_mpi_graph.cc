/*******************************************************************************
 * @file:   mpi_graph_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   30.11.2021
 * @brief:  Unit tests for MPI functions depending on the graph topology.
 ******************************************************************************/
#include "dkaminpar/mpi_graph.h"
#include "dtests/mpi_test.h"

namespace dkaminpar::test {
using namespace fixtures3PE;
// Test sparse all-to-all with one message for each interface node to each adjacent PE
TEST_F(DistributedTriangles, TestInterfaceToPE) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  struct Message {
    PEID from_pe;
    GlobalNodeID from;
    GlobalNodeID to;
    PEID to_pe;
  };

  auto recv_buffers = mpi::graph::sparse_alltoall_interface_to_ghost_get<Message>(
      graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER,
      [&](const NodeID u, const EdgeID, const NodeID v, const PEID pe) -> Message {
        EXPECT_FALSE(graph.is_ghost_node(u));
        EXPECT_TRUE(graph.is_ghost_node(v));
        EXPECT_EQ(graph.ghost_owner(v), pe);
        EXPECT_NE(rank, pe);

        return {
            .from_pe = rank,
            .from = graph.local_to_global_node(u),
            .to = graph.local_to_global_node(v),
            .to_pe = pe,
        };
      });

  EXPECT_EQ(recv_buffers.size(), 3);
  EXPECT_TRUE(recv_buffers[rank].empty());
  for (PEID pe : {0, 1, 2}) {
    if (pe != rank) {
      EXPECT_EQ(recv_buffers[pe].size(), 2);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    for (const auto &message : recv_buffers[pe]) {
      EXPECT_EQ(message.from_pe, pe);
      EXPECT_EQ(message.to_pe, rank);
      EXPECT_TRUE(graph.contains_global_node(message.from));
      EXPECT_TRUE(graph.is_owned_global_node(message.to));

      const NodeID local = graph.global_to_local_node(message.to);
      bool found_from = false;
      for (const auto [e, v] : graph.neighbors(local)) {
        if (message.from == graph.local_to_global_node(v)) {
          found_from = true;
          break;
        }
      }
      EXPECT_TRUE(found_from);
    }
  }
}

// Test sparse all-to-all with one message for each interface node to each adjacent ghost node
TEST_F(DistributedTriangles, TestInterfaceToGhost) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  struct Message {
    PEID from_pe;
    GlobalNodeID from;
    PEID to_pe;
  };

  auto recv_buffers = mpi::graph::sparse_alltoall_interface_to_pe_get<Message>(
      graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER, [&](const NodeID u, const PEID pe) -> Message {
        EXPECT_FALSE(graph.is_ghost_node(u));
        EXPECT_NE(rank, pe);

        return {
            .from_pe = rank,
            .from = graph.local_to_global_node(u),
            .to_pe = pe,
        };
      });

  EXPECT_EQ(recv_buffers.size(), 3);
  EXPECT_TRUE(recv_buffers[rank].empty());
  for (const PEID pe : {0, 1, 2}) {
    if (pe != rank) {
      EXPECT_EQ(recv_buffers[pe].size(), 2);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    for (const auto &message : recv_buffers[pe]) {
      EXPECT_EQ(message.from_pe, pe);
      EXPECT_EQ(message.to_pe, rank);
      EXPECT_TRUE(graph.contains_global_node(message.from));

      bool found_from = false;
      for (const NodeID u : graph.nodes()) {
        for (const auto [e, v] : graph.neighbors(u)) {
          if (message.from == graph.local_to_global_node(v)) {
            found_from = true;
            break;
          }
        }
      }
      EXPECT_TRUE(found_from);
    }
  }
}
} // namespace dkaminpar::test