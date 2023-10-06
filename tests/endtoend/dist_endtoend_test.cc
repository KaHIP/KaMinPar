/*******************************************************************************
 * End-to-end test for the distributed memory library interface.
 *
 * @file:   dist_endtoend_test.cc
 * @author: Daniel Seemaier
 * @date:   06.10.023
 ******************************************************************************/
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
TEST(ShmEndToEndTest, partitions_empty_unweighted_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  std::vector<GlobalNodeID> vtxdist(size + 1);
  std::vector<GlobalEdgeID> xadj{0};
  std::vector<GlobalNodeID> adjncy{};

  std::vector<BlockID> partition{};
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context());
  dist.import_graph(vtxdist.data(), xadj.data(), adjncy.data(), nullptr, nullptr);
  EXPECT_EQ(dist.compute_partition(0, 16, partition.data()), 0);
}

TEST(ShmEndToEndTest, partitions_empty_weighted_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  std::vector<GlobalNodeID> vtxdist(size + 1);
  std::vector<GlobalEdgeID> xadj{0};
  std::vector<GlobalNodeID> adjncy{};
  std::vector<GlobalNodeWeight> vwgt{};
  std::vector<GlobalEdgeWeight> adjwgt{};

  std::vector<BlockID> partition{};
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context());
  dist.import_graph(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
  EXPECT_EQ(dist.compute_partition(0, 16, partition.data()), 0);
}
} // namespace kaminpar::dist

