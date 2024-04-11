/*******************************************************************************
 * End-to-end test for the distributed memory library interface.
 *
 * @file:   dist_endtoend_test.cc
 * @author: Daniel Seemaier
 * @date:   06.10.023
 ******************************************************************************/
#include <numeric>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/math.h"

namespace kaminpar::dist {
namespace data {
static std::vector<GlobalEdgeID> global_xadj = {
#include "data.graph.xadj"
};
static std::vector<GlobalNodeID> global_adjncy = {
#include "data.graph.adjncy"
};

std::vector<GlobalNodeID> create_vtxdist() {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const GlobalNodeID global_n = global_xadj.size() - 1;

  std::vector<GlobalNodeID> vtxdist(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    const auto [from, to] = math::compute_local_range<GlobalNodeID>(global_n, size, pe);
    vtxdist[pe] = from;
    vtxdist[pe + 1] = to;
  }

  return vtxdist;
}

std::vector<GlobalEdgeID> create_xadj() {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const GlobalNodeID global_n = global_xadj.size() - 1;
  const auto [from, to] = math::compute_local_range<GlobalNodeID>(global_n, size, rank);
  const GlobalNodeID n = to - from;

  std::vector<GlobalEdgeID> xadj(n + 1);
  for (GlobalNodeID u = from; u < to; ++u) {
    xadj[u - from] = global_xadj[u] - global_xadj[from];
  }
  xadj[n] = global_xadj[to] - global_xadj[from];

  return xadj;
}
} // namespace data

TEST(DistEndToEndTest, partitions_empty_unweighted_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  std::vector<GlobalNodeID> vtxdist(size + 1);
  std::vector<GlobalEdgeID> xadj{0};
  std::vector<GlobalNodeID> adjncy{};

  std::vector<BlockID> partition{};
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context());
  dist.set_output_level(OutputLevel::QUIET);
  dist.import_graph(vtxdist.data(), xadj.data(), adjncy.data(), nullptr, nullptr);
  EXPECT_EQ(dist.compute_partition(16, partition.data()), 0);
}

TEST(DistEndToEndTest, partitions_empty_weighted_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  std::vector<GlobalNodeID> vtxdist(size + 1);
  std::vector<GlobalEdgeID> xadj{0};
  std::vector<GlobalNodeID> adjncy{};
  std::vector<GlobalNodeWeight> vwgt{};
  std::vector<GlobalEdgeWeight> adjwgt{};

  std::vector<BlockID> partition{};
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context());
  dist.set_output_level(OutputLevel::QUIET);
  dist.import_graph(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.data(), adjwgt.data());
  EXPECT_EQ(dist.compute_partition(16, partition.data()), 0);
}

TEST(DistEndToEndTest, partitions_unweighted_walshaw_data_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  auto vtxdist = data::create_vtxdist();
  auto xadj = data::create_xadj();

  const GlobalNodeID global_n = data::global_xadj.size() - 1;
  const NodeID n = xadj.size() - 1;

  GlobalNodeID *vtxdist_ptr = vtxdist.data();
  GlobalEdgeID *xadj_ptr = xadj.data();
  GlobalNodeID *adjncy_ptr = data::global_adjncy.data() + data::global_xadj[vtxdist[rank]];

  std::vector<BlockID> partition(global_n);
  dKaMinPar::reseed(0);
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context()); // 1 thread: deterministic
  dist.set_output_level(OutputLevel::QUIET);
  dist.import_graph(vtxdist_ptr, xadj_ptr, adjncy_ptr, nullptr, nullptr);
  const EdgeWeight reported_cut = dist.compute_partition(16, partition.data() + vtxdist[rank]);

  // Cut should be around 1200 -- 1300
  EXPECT_LE(reported_cut, 2000);

  // Check that the reported cut matches the partition cut
  std::vector<int> counts(size);
  for (PEID pe = 0; pe < size; ++pe) {
    const auto [from, to] = math::compute_local_range<GlobalNodeID>(global_n, size, pe);
    counts[pe] = static_cast<int>(to - from);
  }

  std::vector<int> displs(size + 1);
  std::partial_sum(counts.begin(), counts.end(), displs.begin() + 1);

  MPI_Allgatherv(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      partition.data(),
      counts.data(),
      displs.data(),
      mpi::type::get<BlockID>(),
      MPI_COMM_WORLD
  );

  EdgeWeight actual_cut = 0;

  for (GlobalNodeID u = 0; u < global_n; ++u) {
    for (GlobalEdgeID e = data::global_xadj[u]; e < data::global_xadj[u + 1]; ++e) {
      const GlobalNodeID v = data::global_adjncy[e];
      if (partition[u] != partition[v]) {
        ++actual_cut;
      }
    }
  }

  EXPECT_EQ(reported_cut, actual_cut / 2);
}

// Disabled: can fail since we offset the PRNG seed by the thread ID, and not all calls are made by
// the same threads across multiple runs
/*
TEST(DistEndToEndTest, partitions_unweighted_walshaw_data_graph_multiple_times_with_same_seed) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  auto vtxdist = data::create_vtxdist();
  auto xadj = data::create_xadj();

  const GlobalNodeID global_n = data::global_xadj.size() - 1;
  const NodeID n = xadj.size() - 1;

  GlobalNodeID *vtxdist_ptr = vtxdist.data();
  GlobalEdgeID *xadj_ptr = xadj.data();
  GlobalNodeID *adjncy_ptr = data::global_adjncy.data() + data::global_xadj[vtxdist[rank]];

  std::vector<BlockID> seed0_partition(n);
  dKaMinPar::reseed(0);
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context()); // 1 thread: deterministic
  dist.set_output_level(OutputLevel::QUIET);
  dist.import_graph(vtxdist_ptr, xadj_ptr, adjncy_ptr, nullptr, nullptr);
  const EdgeWeight reported_cut = dist.compute_partition(16, seed0_partition.data());

  for (const int seed : {0, 0, 0}) {
    std::vector<BlockID> partition(n);
    dKaMinPar::reseed(seed);
    dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context()); // 1 thread: deterministic
    dist.set_output_level(OutputLevel::QUIET);
    dist.import_graph(vtxdist_ptr, xadj_ptr, adjncy_ptr, nullptr, nullptr);
    dist.compute_partition(16, partition.data());
    EXPECT_EQ(partition, seed0_partition);
  }
}
*/

TEST(
    DistEndToEndTest, partitions_unweighted_walshaw_data_graph_multiple_times_with_different_seeds
) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  auto vtxdist = data::create_vtxdist();
  auto xadj = data::create_xadj();

  const GlobalNodeID global_n = data::global_xadj.size() - 1;
  const NodeID n = xadj.size() - 1;

  GlobalNodeID *vtxdist_ptr = vtxdist.data();
  GlobalEdgeID *xadj_ptr = xadj.data();
  GlobalNodeID *adjncy_ptr = data::global_adjncy.data() + data::global_xadj[vtxdist[rank]];

  std::vector<BlockID> seed0_partition(n);
  dKaMinPar::reseed(0);
  dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context()); // 1 thread: deterministic
  dist.set_output_level(OutputLevel::QUIET);
  dist.import_graph(vtxdist_ptr, xadj_ptr, adjncy_ptr, nullptr, nullptr);
  const EdgeWeight reported_cut = dist.compute_partition(16, seed0_partition.data());

  for (const int seed : {1, 2, 3}) {
    std::vector<BlockID> partition(n);
    dKaMinPar::reseed(seed);
    dKaMinPar dist(MPI_COMM_WORLD, 1, create_default_context()); // 1 thread: deterministic
    dist.set_output_level(OutputLevel::QUIET);
    dist.import_graph(vtxdist_ptr, xadj_ptr, adjncy_ptr, nullptr, nullptr);
    dist.compute_partition(16, partition.data());
    EXPECT_NE(partition, seed0_partition);
  }
}
} // namespace kaminpar::dist
