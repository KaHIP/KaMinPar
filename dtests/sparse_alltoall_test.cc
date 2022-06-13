#include <gtest/gtest.h>

#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/wrapper.h"

using namespace testing;

namespace dkaminpar {
TEST(SparseAllToAllTest, regular_alltoall_one_element_works) {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

    // send one message to each PE containing this PEs rank
    std::vector<std::vector<int>> sendbuf(size);
    for (PEID pe = 0; pe < size; ++pe) {
        sendbuf[pe].push_back(rank);
    }

    std::vector<bool> got_message_from(size);

    mpi::sparse_alltoall<int>(
        sendbuf,
        [&](const auto& recvbuf, const PEID from) {
            got_message_from[from] = true;
            EXPECT_EQ(recvbuf.size(), 1);
            EXPECT_EQ(recvbuf.front(), from);
        },
        MPI_COMM_WORLD, true);

    for (PEID pe = 0; pe < size; ++pe) {
        EXPECT_TRUE(got_message_from[pe]);
    }
}

// send single message to next PE
TEST(SparseAllToAllTest, ring_exchange_works) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    const PEID next = (rank + 1) % size;
    const PEID prev = (rank + size - 1) % size;

    std::vector<std::vector<int>> sendbuf(size);
    sendbuf[next].push_back(rank);

    bool got_message_from_prev = false;
    mpi::sparse_alltoall<int>(
        sendbuf,
        [&](const auto& recvbuf, const PEID from) {
            if (from == prev) {
                got_message_from_prev = true;
                EXPECT_EQ(recvbuf.size(), 1);
                EXPECT_EQ(recvbuf.front(), from);
            } else {
                EXPECT_TRUE(recvbuf.empty());
            }
        },
        MPI_COMM_WORLD, true);

    EXPECT_TRUE(got_message_from_prev);
}

// each PE $j sends $i messages to all PEs $i < $j containing $j
TEST(SparseAllToAllTest, triangle_alltoall_works) {
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

    std::vector<bool> got_message_from(size);
    mpi::sparse_alltoall<int>(
        sendbuf,
        [&](const auto& recvbuf, const PEID from) {
            if (from >= rank) {
                got_message_from[from] = true;
                EXPECT_EQ(recvbuf.size(), rank);
                EXPECT_TRUE(
                    std::all_of(recvbuf.begin(), recvbuf.end(), [&](const int value) { return value == from; }));
            } else {
                EXPECT_TRUE(recvbuf.empty());
            }
        },
        MPI_COMM_WORLD, true);

    for (PEID pe = 0; pe < size; ++pe) {
        if (pe >= rank) {
            EXPECT_TRUE(got_message_from[pe]);
        }
    }
}
} // namespace dkaminpar
