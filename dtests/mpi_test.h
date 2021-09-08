#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/mpi_utils.h"

#include <gmock/gmock.h>

namespace dkaminpar::test {
class DistributedGraphFixture : public ::testing::Test {
protected:
  void SetUp() override { std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD); }
  PEID size;
  PEID rank;

  GlobalNodeID next(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u + step) % n;
  }

  GlobalNodeID prev(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u < step) ? n + u - step : u - step;
  }
};
} // namespace dkaminpar::test