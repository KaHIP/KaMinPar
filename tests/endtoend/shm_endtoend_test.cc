/*******************************************************************************
 * End-to-end test for the shared-memory library interface.
 *
 * @file:   shm_endtoend_test.cc
 * @author: Daniel Seemaier
 * @date:   06.10.023
 ******************************************************************************/
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
TEST(ShmEndToEndTest, partitions_empty_unweighted_graph) {
  std::vector<EdgeID> xadj{0};
  std::vector<NodeID> adjncy{};

  { // copy graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.copy_graph(0, xadj.data(), adjncy.data(), nullptr, nullptr);
    EXPECT_EQ(shm.compute_partition(0, 16, partition.data()), 0);
  }

  { // take ownership of graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.take_graph(0, xadj.data(), adjncy.data(), nullptr, nullptr);
    EXPECT_EQ(shm.compute_partition(0, 16, partition.data()), 0);
  }
}

TEST(ShmEndToEndTest, partitions_empty_weighted_graph) {
  std::vector<EdgeID> xadj{0};
  std::vector<NodeID> adjncy{};
  std::vector<NodeWeight> vwgt{};
  std::vector<EdgeWeight> adjwgt{};

  { // copy graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.copy_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(0, 16, partition.data()), 0);
  }

  { // take ownership of graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.take_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(0, 16, partition.data()), 0);
  }
}
} // namespace kaminpar::shm
