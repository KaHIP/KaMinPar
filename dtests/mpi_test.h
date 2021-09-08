#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/mpi_utils.h"

#include <gmock/gmock.h>

namespace dkaminpar::test {
class MpiTestFixture : public ::testing::Test {
protected:
  void SetUp() override { std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD); }
  PEID size;
  PEID rank;
};
} // namespace dkaminpar::test