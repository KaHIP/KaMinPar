#include <gmock/gmock.h>
#include <mpi.h>

#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/sparse_allreduce.h"

using namespace ::testing;

namespace kaminpar::mpi {
template <typename Implementation> struct InplaceSparseAllreduceTest : public ::testing::Test {
  Implementation impl;
};

template <typename T> struct InplaceMPI {
  void operator()(std::vector<T> &buffer, MPI_Comm comm) {
    inplace_sparse_allreduce(tag::mpi_allreduce, buffer, buffer.size(), MPI_SUM, comm);
  }
};

template <typename T> struct InplaceDoubling {
  void operator()(std::vector<T> &buffer, MPI_Comm comm) {
    inplace_sparse_allreduce(tag::doubling_allreduce, buffer, buffer.size(), MPI_SUM, comm);
  }
};

template <typename T>
using InplaceSparseAllreduceImplementations = ::testing::Types<InplaceMPI<T>, InplaceDoubling<T>>;
TYPED_TEST_SUITE(InplaceSparseAllreduceTest, InplaceSparseAllreduceImplementations<int>);

TYPED_TEST(InplaceSparseAllreduceTest, empty_allreduce) {
  std::vector<int> buf;
  this->impl(buf, MPI_COMM_WORLD);
  EXPECT_TRUE(buf.empty());
}

TYPED_TEST(InplaceSparseAllreduceTest, one) {
  std::vector<int> buf(1, 1);
  this->impl(buf, MPI_COMM_WORLD);
  EXPECT_EQ(buf.size(), 1);
  EXPECT_EQ(buf.front(), get_comm_size(MPI_COMM_WORLD));
}

TYPED_TEST(InplaceSparseAllreduceTest, one_per_pe) {
  const PEID size = get_comm_size(MPI_COMM_WORLD);
  const PEID rank = get_comm_rank(MPI_COMM_WORLD);

  std::vector<int> buf(size);
  buf[rank] = 1;
  this->impl(buf, MPI_COMM_WORLD);

  EXPECT_EQ(buf.size(), size);
  EXPECT_THAT(buf, Each(Eq(1)));
}
} // namespace kaminpar::mpi
