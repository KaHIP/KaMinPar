#include <gtest/gtest.h>

#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/wrapper.h"

using namespace testing;

namespace dkaminpar {
template <typename Implementation>
struct SparseAlltoallTest : public Test {
    Implementation impl;
};

template <typename T>
struct CompleteSendRecvImplementation {
    std::vector<std::vector<T>> operator()(const std::vector<std::vector<T>>& sendbuf, MPI_Comm comm, bool self) {
        std::vector<std::vector<T>> recvbufs(mpi::get_comm_size(comm));
        mpi::sparse_alltoall_complete<T, std::vector<T>>(
            sendbuf, [&](auto recvbuf, const PEID pe) { recvbufs[pe] = std::move(recvbuf); }, self, comm);
        return recvbufs;
    }
};

template <typename T>
using SparseAlltoallImplementations = Types<CompleteSendRecvImplementation<T>>;

TYPED_TEST_SUITE(SparseAlltoallTest, SparseAlltoallImplementations<int>);

TYPED_TEST(SparseAlltoallTest, regular_single_element_alltoall) {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

    // send one message to each PE containing this PEs rank
    std::vector<std::vector<int>> sendbuf(size);
    for (PEID pe = 0; pe < size; ++pe) {
        sendbuf[pe].push_back(rank);
    }

    auto recvbufs = this->impl(sendbuf, MPI_COMM_WORLD, true);

    for (PEID from = 0; from < size; ++from) {
        EXPECT_EQ(recvbufs[from].size(), 1);
        EXPECT_EQ(recvbufs[from].front(), from);
    }
}

// send single message to next PE
TYPED_TEST(SparseAlltoallTest, ring_exchange) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    const PEID next = (rank + 1) % size;
    const PEID prev = (rank + size - 1) % size;

    std::vector<std::vector<int>> sendbuf(size);
    sendbuf[next].push_back(rank);

    auto recvbufs = this->impl(sendbuf, MPI_COMM_WORLD, true);

    for (PEID from = 0; from < size; ++from) {
        if (from == prev) {
            EXPECT_EQ(recvbufs[from].size(), 1);
            EXPECT_EQ(recvbufs[from].front(), from);
        } else {
            EXPECT_TRUE(recvbufs[from].empty());
        }
    }
}

// each PE $j sends $i messages to all PEs $i < $j containing $j
TYPED_TEST(SparseAlltoallTest, irregular_triangle_alltoall) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

    std::vector<std::vector<int>> sendbuf(size);
    for (PEID pe = 0; pe < size; ++pe) {
        if (pe <= rank) {
            for (int i = 0; i < pe; ++i) {
                sendbuf[pe].push_back(rank);
            }
        }
    }

    auto recvbufs = this->impl(sendbuf, MPI_COMM_WORLD, true);

    for (PEID from = 0; from < size; ++from) {
        if (from >= rank) {
            EXPECT_EQ(recvbufs[from].size(), rank);
            EXPECT_TRUE(std::all_of(
                recvbufs[from].begin(), recvbufs[from].end(), [&](const int value) { return value == from; }));
        } else {
            EXPECT_TRUE(recvbufs[from].empty());
        }
    }
}

TEST(DefaultSparseAlltoallTest, does_not_move_lvalue_reference) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

    std::vector<std::vector<int>> sendbuf(size);
    for (PEID pe = 0; pe < size; ++pe) {
        sendbuf[pe].push_back(pe);
    }

    mpi::sparse_alltoall<int>(
        sendbuf, [&](auto) {}, true, MPI_COMM_WORLD);

    EXPECT_EQ(sendbuf.size(), size);
    for (PEID pe = 0; pe < size; ++pe) {
        EXPECT_EQ(sendbuf[pe].size(), 1);
        EXPECT_EQ(sendbuf[pe].front(), pe);
    }
}
} // namespace dkaminpar
