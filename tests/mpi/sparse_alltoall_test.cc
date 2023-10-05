#include <gtest/gtest.h>

#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/sparse_alltoall.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-common/math.h"

namespace kaminpar::mpi {
template <typename Implementation> struct SparseAlltoallTest : public ::testing::Test {
  Implementation impl;
};

template <typename T> struct AlltoallvImplementation {
  std::vector<std::vector<T>>
  operator()(const std::vector<std::vector<T>> &sendbuf, MPI_Comm comm) {
    std::vector<std::vector<T>> recvbufs(get_comm_size(comm));
    sparse_alltoall<T, std::vector<T>>(
        tag::alltoallv,
        sendbuf,
        [&](auto recvbuf, const PEID pe) { recvbufs[pe] = std::move(recvbuf); },
        comm
    );
    return recvbufs;
  }
};

template <typename T> struct GridImplementation {
  std::vector<std::vector<T>>
  operator()(const std::vector<std::vector<T>> &sendbuf, MPI_Comm comm) {
    const PEID size = get_comm_size(comm);

    std::vector<std::vector<T>> recvbufs(size);
    sparse_alltoall<T, std::vector<T>>(
        tag::grid,
        sendbuf,
        [&](auto recvbuf, const PEID pe) { recvbufs[pe] = std::move(recvbuf); },
        comm
    );
    return recvbufs;
  }
};

template <typename T> struct CompleteSendRecvImplementation {
  std::vector<std::vector<T>>
  operator()(const std::vector<std::vector<T>> &sendbuf, MPI_Comm comm) {
    std::vector<std::vector<T>> recvbufs(get_comm_size(comm));
    sparse_alltoall<T, std::vector<T>>(
        tag::complete_send_recv,
        sendbuf,
        [&](auto recvbuf, const PEID pe) { recvbufs[pe] = std::move(recvbuf); },
        comm
    );
    return recvbufs;
  }
};

template <typename T>
using SparseAlltoallImplementations = ::testing::
    Types<CompleteSendRecvImplementation<T>, AlltoallvImplementation<T>, GridImplementation<T>>;
TYPED_TEST_SUITE(SparseAlltoallTest, SparseAlltoallImplementations<int>);

TYPED_TEST(SparseAlltoallTest, empty_alltoall) {
  const auto [size, rank] = get_comm_info(MPI_COMM_WORLD);

  std::vector<std::vector<int>> sendbufs(size);
  const auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);

  ASSERT_EQ(recvbufs.size(), size);
  for (PEID pe = 0; pe < size; ++pe) {
    EXPECT_TRUE(recvbufs[pe].empty());
  }
}

TYPED_TEST(SparseAlltoallTest, full_alltoall) {
  const int num_elements = 10;
  const auto [size, rank] = get_comm_info(MPI_COMM_WORLD);

  std::vector<std::vector<int>> sendbufs(size);
  for (PEID pe = 0; pe < size; ++pe) {
    for (int i = 0; i < num_elements; ++i) {
      sendbufs[pe].push_back(rank + i);
    }
  }

  auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);

  ASSERT_EQ(recvbufs.size(), size);
  for (PEID pe = 0; pe < size; ++pe) {
    ASSERT_EQ(recvbufs[pe].size(), num_elements);

    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(recvbufs[pe][i], pe + i);
    }
  }
}

TYPED_TEST(SparseAlltoallTest, single_message) {
  const auto [size, rank] = get_comm_info(MPI_COMM_WORLD);

  std::vector<std::vector<int>> sendbufs(size);
  if (rank == 0) {
    sendbufs[size - 1].push_back(42);
  }

  auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);

  if (rank == size - 1) {
    ASSERT_EQ(recvbufs[0].size(), 1);
    EXPECT_EQ(recvbufs[0].front(), 42);
  } else {
    EXPECT_TRUE(recvbufs[0].empty());
  }

  for (PEID pe = 1; pe < size; ++pe) {
    EXPECT_TRUE(recvbufs[pe].empty());
  }
}

TYPED_TEST(SparseAlltoallTest, regular_single_element_alltoall) {
  const auto [size, rank] = get_comm_info(MPI_COMM_WORLD);

  // send one message to each PE containing this PEs rank
  std::vector<std::vector<int>> sendbufs(size);
  for (PEID pe = 0; pe < size; ++pe) {
    sendbufs[pe].push_back(rank);
  }

  auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);
  ASSERT_EQ(recvbufs.size(), size);

  for (PEID from = 0; from < size; ++from) {
    ASSERT_EQ(recvbufs[from].size(), 1);
    EXPECT_EQ(recvbufs[from].front(), from);
  }
}

// send single message to next PE
TYPED_TEST(SparseAlltoallTest, ring_exchange) {
  const PEID size = get_comm_size(MPI_COMM_WORLD);
  const PEID rank = get_comm_rank(MPI_COMM_WORLD);
  const PEID next = (rank + 1) % size;
  const PEID prev = (rank + size - 1) % size;

  std::vector<std::vector<int>> sendbufs(size);
  sendbufs[next].push_back(rank);

  auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);
  ASSERT_EQ(recvbufs.size(), size);

  for (PEID from = 0; from < size; ++from) {
    if (from == prev) {
      ASSERT_EQ(recvbufs[from].size(), 1);
      EXPECT_EQ(recvbufs[from].front(), from);
    } else {
      EXPECT_TRUE(recvbufs[from].empty());
    }
  }
}

// each PE $j sends $i messages to all PEs $i < $j containing $j
TYPED_TEST(SparseAlltoallTest, irregular_triangle_alltoall) {
  const PEID size = get_comm_size(MPI_COMM_WORLD);
  const PEID rank = get_comm_rank(MPI_COMM_WORLD);

  std::vector<std::vector<int>> sendbufs(size);
  for (PEID pe = 0; pe < size; ++pe) {
    if (pe <= rank) {
      for (int i = 0; i < pe; ++i) {
        sendbufs[pe].push_back(rank);
      }
    }
  }

  auto recvbufs = this->impl(sendbufs, MPI_COMM_WORLD);

  for (PEID from = 0; from < size; ++from) {
    if (from >= rank) {
      EXPECT_EQ(recvbufs[from].size(), rank);
      EXPECT_TRUE(std::all_of(recvbufs[from].begin(), recvbufs[from].end(), [&](const int value) {
        return value == from;
      }));
    } else {
      EXPECT_TRUE(recvbufs[from].empty());
    }
  }
}

TEST(DefaultSparseAlltoallTest, does_not_move_lvalue_reference) {
  const PEID size = get_comm_size(MPI_COMM_WORLD);

  std::vector<std::vector<int>> sendbufs(size);
  for (PEID pe = 0; pe < size; ++pe) {
    sendbufs[pe].push_back(pe);
  }

  sparse_alltoall<int>(
      sendbufs, [&](auto) {}, MPI_COMM_WORLD
  );

  EXPECT_EQ(sendbufs.size(), size);
  for (PEID pe = 0; pe < size; ++pe) {
    EXPECT_EQ(sendbufs[pe].size(), 1);
    EXPECT_EQ(sendbufs[pe].front(), pe);
  }
}
} // namespace kaminpar::mpi
