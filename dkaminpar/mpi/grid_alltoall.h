/*******************************************************************************
 * @file:   grid_alltoall.h
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communication.
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <unordered_map>

#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/grid_topology.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/logger.h"
#include "common/math.h"
#include "common/noinit_vector.h"
#include "common/parallel/algorithm.h"
#include "common/preallocated_vector.h"
#include "common/timer.h"

namespace kaminpar::mpi {
namespace internal {
class GridCommunicator {
public:
    GridCommunicator(MPI_Comm comm) {
        const auto [size, rank] = get_comm_info(comm);
        GridTopology topo(size);
        MPI_Comm_split(comm, topo.row(rank), rank, &_row_comm);
        MPI_Comm_split(
            comm, topo.virtual_col(rank), topo.virtual_col(rank) == topo.col(rank) ? rank : size + rank, &_col_comm
        );
    }

    MPI_Comm row_comm() const {
        return _row_comm;
    }

    PEID row_comm_size() const {
        return get_comm_size(_row_comm);
    }

    MPI_Comm col_comm() const {
        return _col_comm;
    }

    PEID col_comm_size() const {
        return get_comm_size(_col_comm);
    }

private:
    MPI_Comm _row_comm;
    MPI_Comm _col_comm;
};
} // namespace internal

template <typename Message, typename Buffer, typename SendBuffer, typename CountsBuffer, typename Receiver>
void sparse_alltoall_grid(SendBuffer&& data, const CountsBuffer& counts, Receiver&& receiver, MPI_Comm comm) {
    using namespace internal;

    static std::unordered_map<MPI_Comm, GridCommunicator> grid_communicators;
    auto [grid_comm_it, ignored] = grid_communicators.try_emplace(comm, comm);
    GridCommunicator& grid_comm  = grid_comm_it->second;

    const auto& row_comm      = grid_comm.row_comm();
    const PEID  row_comm_size = grid_comm.row_comm_size();
    const auto& col_comm      = grid_comm.col_comm();
    const PEID  col_comm_size = grid_comm.col_comm_size();

    const PEID size = mpi::get_comm_size(comm);
    const PEID rank = mpi::get_comm_rank(comm);

    GridTopology topo(size);
    const PEID   my_row         = topo.row(rank);
    const PEID   my_col         = topo.col(rank);
    const PEID   my_virtual_col = topo.virtual_col(rank);

    KASSERT(row_comm_size == topo.row_size(my_row));
    KASSERT(col_comm_size == topo.virtual_col_size(my_virtual_col));

    /*
     * Step 1 (column)
     *
     * For each row, send counts for all PEs in that row to the PE in our column in that row
     */

    //START_TIMER("First hop allocation");
    /*static*/ std::vector<int> row_counts_send_counts;
    /*static*/ std::vector<int> row_counts_recv_counts;
    /*static*/ std::vector<int> row_counts_send_displs;
    /*static*/ std::vector<int> row_counts_recv_displs;
    if (row_counts_send_counts.empty()) {
        // Compute send counts
        row_counts_send_counts.resize(col_comm_size);
        for (PEID row = 0; row < topo.num_rows(); ++row) {
            row_counts_send_counts[row] = topo.num_cols_in_row(row);
        }

        // Compute recv counts
        row_counts_recv_counts.resize(col_comm_size);
        if (my_col == my_virtual_col) {
            std::fill(row_counts_recv_counts.begin(), row_counts_recv_counts.end(), topo.row_size(my_row));
        }
        // else {
        //     std::fill(row_counts_recv_counts.begin(), row_counts_recv_counts.end(), 0);
        // }

        // Compute buffer displacements
        row_counts_send_displs.resize(col_comm_size + 1);
        row_counts_recv_displs.resize(col_comm_size + 1);
        std::partial_sum(
            row_counts_send_counts.begin(), row_counts_send_counts.end(), row_counts_send_displs.begin() + 1
        );
        std::partial_sum(
            row_counts_recv_counts.begin(), row_counts_recv_counts.end(), row_counts_recv_displs.begin() + 1
        );
    }

    std::vector<int> row_counts(row_comm_size * col_comm_size);
    //STOP_TIMER();

    // Exchange counts within payload
    {
        //SCOPED_TIMER("First hop counts MPI_Alltoallv");

        KASSERT(asserting_cast<int>(row_counts.size()) >= row_counts_recv_displs.back());
        KASSERT(asserting_cast<int>(row_counts_send_counts.size()) >= col_comm_size);
        KASSERT(asserting_cast<int>(counts.size()) >= row_counts_send_displs.back());
        KASSERT(asserting_cast<int>(row_counts_recv_counts.size()) >= col_comm_size);
        KASSERT(asserting_cast<int>(row_counts_recv_displs.size()) >= col_comm_size);
        KASSERT(asserting_cast<int>(row_counts_send_displs.size()) >= col_comm_size);

        mpi::alltoallv(
            counts.data(), row_counts_send_counts.data(), row_counts_send_displs.data(), row_counts.data(),
            row_counts_recv_counts.data(), row_counts_recv_displs.data(), col_comm
        );
    }

    /*
     * After Step 1 (column)
     *
     * row_counts contains data counts for PEs in my row:
     * from PE in row 0: data count for col0, col1, col2, ...,  (row_comm_size items)
     * from PE in row 1: data count for col0, col1, col2, ...,  (row_comm_size items)
     * ...
     *
     * Size: row_comm_size * col_comm_size
     */

    /*
     * Step 2 (column)
     *
     * Exchange data within our column: send PE in row 0 all data for row 0, ...
     */

    //START_TIMER("First hop allocation");
    std::vector<int> row_send_counts(col_comm_size);
    std::vector<int> row_recv_counts(col_comm_size);
    std::vector<int> row_send_displs(col_comm_size + 1);
    std::vector<int> row_recv_displs(col_comm_size + 1);
    std::vector<int> row_displs(row_comm_size * col_comm_size + 1);
    //STOP_TIMER();

    {
        //SCOPED_TIMER("First hop counts summation");

        for (PEID pe = 0; pe < size; ++pe) {
            row_send_counts[topo.row(pe)] += counts[pe];
        }

        tbb::parallel_for<PEID>(0, col_comm_size, [&](const PEID row) {
            row_recv_counts[row] = std::accumulate(
                row_counts.begin() + row * row_comm_size, row_counts.begin() + (row + 1) * row_comm_size, 0
            );
        });

        parallel::prefix_sum(row_send_counts.begin(), row_send_counts.end(), row_send_displs.begin() + 1);
        parallel::prefix_sum(row_recv_counts.begin(), row_recv_counts.end(), row_recv_displs.begin() + 1);
        parallel::prefix_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);
    }

    //START_TIMER("First hop allocation");
    std::vector<Message> row_recv_buf(row_recv_displs.back());
    //STOP_TIMER();

    {
        //SCOPED_TIMER("First hop payload MPI_Alltoallv");

        KASSERT(asserting_cast<PEID>(row_send_counts.size()) >= col_comm_size);
        KASSERT(asserting_cast<PEID>(row_send_displs.size()) >= col_comm_size);
        KASSERT(asserting_cast<int>(data.size()) >= row_send_displs.back());
        KASSERT(asserting_cast<PEID>(row_recv_counts.size()) >= col_comm_size);
        KASSERT(asserting_cast<PEID>(row_recv_displs.size()) >= col_comm_size);
        KASSERT(asserting_cast<int>(row_recv_buf.size()) >= row_recv_displs.back());

        mpi::alltoallv(
            data.data(), row_send_counts.data(), row_send_displs.data(), row_recv_buf.data(), row_recv_counts.data(),
            row_recv_displs.data(), col_comm
        );
    }

    /*
     * After step 2 (column)
     *
     * row_recv_buf contains data for PEs in my row:
     * data from PEs in row 0 for: col0, col1, col2, ...,  (row_comm_size items)
     * data from PEs in row 1 for: col0, col1, col2, ...,  (row_comm_size items)
     * ...                                                 (column_comm_size rows)
     *
     * The data count is determined by row_coutns:
     * data counts from PEs in row0: col0, col1, col2, ...,
     * data counts from PEs in row1: col0, col1, col2, ...,
     * ...
     *
     * Size: row_comm_size * col_comm_size
     */

    /*
     * Step 3 (row)
     *
     * Send each column the data counts from each row
     */

    //START_TIMER("Second hop allocation");
    std::vector<int> col_counts(row_comm_size);
    std::vector<int> col_subcounts(row_comm_size * col_comm_size);
    std::vector<int> subcounts(size);
    std::vector<int> col_recv_counts(row_comm_size);
    std::vector<int> col_recv_displs(row_comm_size + 1);
    std::vector<int> col_displs(row_comm_size + 1);
    //STOP_TIMER();

    {
        //SCOPED_TIMER("Second hop counts summation");

        for (PEID col = 0; col < row_comm_size; ++col) {
            for (PEID from_row = 0; from_row < col_comm_size; ++from_row) {
                const PEID pe = col + from_row * row_comm_size;
                KASSERT(col < asserting_cast<PEID>(col_counts.size()));
                KASSERT(pe < asserting_cast<PEID>(row_counts.size()));
                KASSERT(from_row + col * col_comm_size < asserting_cast<PEID>(col_subcounts.size()));

                col_counts[col] += row_counts[pe];
                col_subcounts[from_row + col * col_comm_size] = row_counts[pe];
            }
        }
    }

    //START_TIMER("Second hop allocation");
    /*static*/ std::vector<int> col_subcounts_send_counts;
    /*static*/ std::vector<int> col_subcounts_send_displs;
    /*static*/ std::vector<int> col_subcounts_recv_counts;
    /*static*/ std::vector<int> col_subcounts_recv_displs;
    if (col_subcounts_send_counts.empty()) {
        col_subcounts_send_counts.resize(row_comm_size);
        if (topo.col(rank) == topo.virtual_col(rank)) {
            std::fill(col_subcounts_send_counts.begin(), col_subcounts_send_counts.end(), col_comm_size);
        }

        col_subcounts_recv_counts.resize(row_comm_size);
        for (PEID col = 0; col < topo.num_full_cols(); ++col) {
            KASSERT(col < asserting_cast<PEID>(col_subcounts_recv_counts.size()));
            col_subcounts_recv_counts[col] = topo.virtual_col_size(col);
        }

        col_subcounts_send_displs.resize(row_comm_size + 1);
        col_subcounts_recv_displs.resize(row_comm_size + 1);
        std::partial_sum(
            col_subcounts_send_counts.begin(), col_subcounts_send_counts.end(), col_subcounts_send_displs.begin() + 1
        );
        std::partial_sum(
            col_subcounts_recv_counts.begin(), col_subcounts_recv_counts.end(), col_subcounts_recv_displs.begin() + 1
        );
    }
    //STOP_TIMER();

    // Exchange counts
    {
        //SCOPED_TIMER("Second hop counts MPI_Alltoallv");

        KASSERT(asserting_cast<PEID>(col_subcounts_send_counts.size()) >= row_comm_size);
        KASSERT(asserting_cast<PEID>(col_subcounts_recv_counts.size()) >= row_comm_size);
        KASSERT(asserting_cast<int>(col_subcounts.size()) >= col_subcounts_send_displs.back());
        KASSERT(asserting_cast<PEID>(col_subcounts_recv_counts.size()) >= row_comm_size);
        KASSERT(asserting_cast<PEID>(col_subcounts_recv_displs.size()) >= row_comm_size);
        KASSERT(asserting_cast<int>(subcounts.size()) >= col_subcounts_recv_displs.back());

        mpi::alltoallv(
            col_subcounts.data(), col_subcounts_send_counts.data(), col_subcounts_send_displs.data(), subcounts.data(),
            col_subcounts_recv_counts.data(), col_subcounts_recv_displs.data(), row_comm
        );
    }

    {
        //SCOPED_TIMER("Second hop counts summation");

        PEID pe = 0;
        for (PEID row = 0; row < topo.num_full_cols(); ++row) {
            int sum = 0;

            for (PEID col = 0; col < topo.virtual_col_size(row); ++col) {
                KASSERT(pe < asserting_cast<PEID>(subcounts.size()));
                sum += subcounts[pe++];
            }

            KASSERT(row < asserting_cast<PEID>(col_recv_counts.size()));
            col_recv_counts[row] = sum;
        }
    }

    // Transpose data buffer
    //START_TIMER("Second hop allocation");
    std::vector<Message> col_data(row_recv_buf.size());
    //STOP_TIMER();

    {
        //SCOPED_TIMER("Second hop data transposition");

        parallel::prefix_sum(col_recv_counts.begin(), col_recv_counts.end(), col_recv_displs.begin() + 1);
        parallel::prefix_sum(col_counts.begin(), col_counts.end(), col_displs.begin() + 1);

        for (PEID col = 0; col < row_comm_size; ++col) {
            KASSERT(col < asserting_cast<PEID>(col_displs.size()));
            int i = col_displs[col];

            for (PEID row = 0; row < col_comm_size; ++row) {
                const PEID pe = row * row_comm_size + col;

                KASSERT(pe < asserting_cast<PEID>(row_displs.size()));
                KASSERT(pe < asserting_cast<PEID>(row_counts.size()));

                const int row_displ = row_displs[pe];
                const int row_count = row_counts[pe];

                KASSERT(row_displ <= asserting_cast<PEID>(row_recv_buf.size()));
                KASSERT(row_displ + row_count <= asserting_cast<PEID>(row_recv_buf.size()));
                KASSERT(i <= asserting_cast<int>(col_data.size()));
                KASSERT(i + row_count <= asserting_cast<int>(col_data.size()));

                std::copy(
                    row_recv_buf.begin() + row_displ, row_recv_buf.begin() + row_displ + row_count, col_data.begin() + i
                );
                i += row_count;
            }
        }
    }

    // Exchange col payload
    //START_TIMER("Second hop allocation");
    std::vector<Message> col_recv_buf(col_recv_displs.back());
    //STOP_TIMER();

    {
        //SCOPED_TIMER("Second hop payload MPI_Alltoallv");

        KASSERT(asserting_cast<PEID>(col_counts.size()) >= row_comm_size);
        KASSERT(asserting_cast<PEID>(col_displs.size()) >= row_comm_size);
        KASSERT(asserting_cast<int>(col_data.size()) >= col_displs.back());
        KASSERT(asserting_cast<PEID>(col_recv_counts.size()) >= row_comm_size);
        KASSERT(asserting_cast<PEID>(col_recv_displs.size()) >= row_comm_size);
        KASSERT(asserting_cast<int>(col_recv_buf.size()) >= col_recv_displs.back());

        mpi::alltoallv(
            col_data.data(), col_counts.data(), col_displs.data(), col_recv_buf.data(), col_recv_counts.data(),
            col_recv_displs.data(), row_comm
        );
    }

    // Assertion:
    // col_recv_buf contains all data adressed to this PE:
    // data from 0x0, 1x0, 2x0, ..., 0x1, 0x2, ... ...

    //START_TIMER("Invoke receiver lambda");
    std::size_t displ = 0;
    std::size_t index = 0;
    for (PEID col = 0; col < topo.num_full_cols(); ++col) {
        for (PEID row = 0; row < topo.virtual_col_size(col); ++row) {
            const PEID pe = topo.virtual_element(row, col); // row * col_comm_size + col;
            KASSERT(index < subcounts.size());
            const auto buf_size = subcounts[index++];

            Buffer buffer(buf_size);
            tbb::parallel_for<std::size_t>(0, buf_size, [&](const std::size_t i) {
                KASSERT(i < buffer.size());
                KASSERT(displ + i < col_recv_buf.size());
                buffer[i] = col_recv_buf[displ + i];
            });
            displ += buf_size;

            KASSERT(pe < size);
            invoke_receiver(std::move(buffer), pe, receiver);
        }
    }
    //STOP_TIMER();
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_grid(SendBuffers&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    // @todo avoid using this variant since it requires a full copy of the send buffers

    const auto [size, rank] = mpi::get_comm_info(comm);
    NoinitVector<int> counts(size);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { counts[pe] = asserting_cast<int>(send_buffers[pe].size()); });
    NoinitVector<int> displs(size + 1);
    parallel::prefix_sum(counts.begin(), counts.end(), displs.begin() + 1);
    displs.front() = 0;

    NoinitVector<Message> dense_buffer(displs.back());
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        std::copy(send_buffers[pe].begin(), send_buffers[pe].end(), dense_buffer.begin() + displs[pe]);
    });

    sparse_alltoall_grid<Message, Buffer>(std::move(dense_buffer), counts, std::forward<Receiver>(receiver), comm);
}
} // namespace kaminpar::mpi
