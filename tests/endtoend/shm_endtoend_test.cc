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
namespace data {
static std::vector<EdgeID> xadj = {
#include "data.graph.xadj"
};
static std::vector<NodeID> adjncy = {
#include "data.graph.adjncy"
};
} // namespace data

TEST(ShmEndToEndTest, partitions_empty_unweighted_graph) {
  std::vector<EdgeID> xadj{0};
  std::vector<NodeID> adjncy{};

  { // copy graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.set_output_level(OutputLevel::QUIET);
    shm.copy_graph(0, xadj.data(), adjncy.data(), nullptr, nullptr);
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }

  { // take ownership of graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.set_output_level(OutputLevel::QUIET);
    shm.borrow_and_mutate_graph(0, xadj.data(), adjncy.data(), nullptr, nullptr);
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
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
    shm.set_output_level(OutputLevel::QUIET);
    shm.copy_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }

  { // take ownership of graph
    std::vector<BlockID> partition{};
    KaMinPar shm(4, create_default_context());
    shm.set_output_level(OutputLevel::QUIET);
    shm.borrow_and_mutate_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }
}

TEST(ShmEndToEndTest, partitions_empty_graph_repeatedly_with_separate_partitioner_instances) {
  for (const int seed : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    std::vector<EdgeID> xadj{0};
    std::vector<NodeID> adjncy{};
    std::vector<NodeWeight> vwgt{};
    std::vector<EdgeWeight> adjwgt{};
    std::vector<BlockID> partition{};

    KaMinPar::reseed(seed);
    KaMinPar shm(4, create_default_context());
    shm.set_output_level(OutputLevel::QUIET);
    shm.borrow_and_mutate_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }
}

TEST(ShmEndToEndTest, partitions_empty_graph_repeatedly_with_one_partitioner_instances) {
  KaMinPar shm(4, create_default_context());
  shm.set_output_level(OutputLevel::QUIET);

  for (const int seed : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    std::vector<EdgeID> xadj{0};
    std::vector<NodeID> adjncy{};
    std::vector<NodeWeight> vwgt{};
    std::vector<EdgeWeight> adjwgt{};
    std::vector<BlockID> partition{};

    KaMinPar::reseed(seed);
    shm.borrow_and_mutate_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }
}

TEST(ShmEndToEndTest, partitions_empty_graph_repeatedly_after_borrow) {
  std::vector<EdgeID> xadj{0};
  std::vector<NodeID> adjncy{};
  std::vector<NodeWeight> vwgt{};
  std::vector<EdgeWeight> adjwgt{};
  std::vector<BlockID> partition{};

  KaMinPar shm(4, create_default_context());
  shm.borrow_and_mutate_graph(0, xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
  shm.set_output_level(OutputLevel::QUIET);

  for (const int seed : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    KaMinPar::reseed(seed);
    EXPECT_EQ(shm.compute_partition(16, partition.data()), 0);
  }
}

TEST(ShmEndToEndTest, partitions_unweighted_walshaw_data_graph) {
  auto &xadj = data::xadj;
  auto &adjncy = data::adjncy;
  const NodeID n = xadj.size() - 1;

  EdgeWeight reported_cut = 0;

  { // Copy graph
    std::vector<BlockID> partition(n);
    KaMinPar::reseed(0);
    KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
    shm.set_output_level(OutputLevel::QUIET);
    shm.copy_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);
    reported_cut = shm.compute_partition(16, partition.data());

    // Cut should be around 1200 -- 1300
    EXPECT_LE(reported_cut, 2000);

    // Make sure that the reported cut matches the resulting partition
    EdgeWeight expected_cut = 0;
    for (NodeID u = 0; u < n; ++u) {
      for (EdgeID e = xadj[u]; e < xadj[u + 1]; ++e) {
        if (partition[u] != partition[adjncy[e]]) {
          ++expected_cut;
        }
      }
    }
    EXPECT_EQ(reported_cut, expected_cut / 2);
  }

  { // Take graph
    std::vector<BlockID> partition(n);
    KaMinPar::reseed(0);
    KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
    shm.set_output_level(OutputLevel::QUIET);
    shm.borrow_and_mutate_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);

    // Cut should be the same as before
    EXPECT_EQ(shm.compute_partition(16, partition.data()), reported_cut);
  }
}

TEST(ShmEndToEndTest, partitions_unweighted_walshaw_graph_multiple_times_with_same_seed) {
  auto &xadj = data::xadj;
  auto &adjncy = data::adjncy;
  const NodeID n = xadj.size() - 1;

  std::vector<BlockID> seed0_partition(n);
  KaMinPar::reseed(0);
  KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
  shm.set_output_level(OutputLevel::QUIET);
  shm.copy_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);
  shm.compute_partition(16, seed0_partition.data());
  EdgeWeight expected_cut = 0;

  // Partition with the same seed multiple times: result should stay the same
  for (const int seed : {0, 0, 0}) {
    std::vector<BlockID> partition(n);
    KaMinPar::reseed(seed);
    KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
    shm.set_output_level(OutputLevel::QUIET);
    shm.copy_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);
    shm.compute_partition(16, partition.data());
    EXPECT_EQ(seed0_partition, partition);
  }
}

TEST(ShmEndToEndTest, partitions_unweighted_walshaw_graph_multiple_times_with_different_seeds) {
  auto &xadj = data::xadj;
  auto &adjncy = data::adjncy;
  const NodeID n = xadj.size() - 1;

  std::vector<BlockID> seed0_partition(n);
  KaMinPar::reseed(0);
  KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
  shm.set_output_level(OutputLevel::QUIET);
  shm.copy_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);
  shm.compute_partition(16, seed0_partition.data());
  EdgeWeight expected_cut = 0;

  // Partition with the different seeds: result should change
  for (const int seed : {1, 2, 3}) {
    std::vector<BlockID> partition(n);
    KaMinPar::reseed(seed);
    KaMinPar shm(1, create_default_context()); // 1 thread: deterministic
    shm.set_output_level(OutputLevel::QUIET);
    shm.copy_graph(n, xadj.data(), adjncy.data(), nullptr, nullptr);
    shm.compute_partition(16, partition.data());
    EXPECT_NE(seed0_partition, partition);
  }
}
} // namespace kaminpar::shm
