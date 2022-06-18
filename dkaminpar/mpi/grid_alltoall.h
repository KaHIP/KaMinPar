/*******************************************************************************
 * @file:   grid_alltoall.h
 *
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communications.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include <tbb/parallel_for.h>

#include "common/utils/math.h"
#include "common/utils/noinit_vector.h"
#include "common/utils/preallocated_vector.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/wrapper.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar::mpi {
namespace internal {
class GridCommunicator {
public:
    GridCommunicator(const PEID size, const PEID rank, MPI_Comm comm) : _sqrt(static_cast<PEID>(std::sqrt(size))) {
        PEID row = rank / _sqrt;
        PEID col = rank % _sqrt;
        MPI_Comm_split(comm, row, rank, &_row_comm);
        MPI_Comm_split(comm, col, rank, &_col_comm);
    }

    PEID sqrt() const {
        return _sqrt;
    }

    MPI_Comm row_comm() const {
        return _row_comm;
    }

    MPI_Comm col_comm() const {
        return _col_comm;
    }

private:
    PEID     _sqrt;
    MPI_Comm _row_comm;
    MPI_Comm _col_comm;
};
} // namespace internal

template <typename Message, typename Buffer, typename SendBuffer, typename CountsBuffer, typename Receiver>
void sparse_alltoall_grid(SendBuffer&& data, const CountsBuffer& counts, Receiver&& receiver, MPI_Comm comm) {
    using namespace internal;
    using namespace kaminpar;

    const auto [size, rank] = mpi::get_comm_info(comm);
    KASSERT(math::is_square(size), "", assert::always);
    static GridCommunicator grid_comm(size, rank, comm);
    const PEID              num_rows_cols = grid_comm.sqrt();

    SET_DEBUG(false);
    //DBG << V(num_rows_cols);

    //
    // Step 1: Send rows to the right PE in the same column
    //

    // --> Build counts for first hop
    std::vector<int> row_send_counts(num_rows_cols);
    for (PEID row = 0; row < num_rows_cols; ++row) {
        for (PEID col = 0; col < num_rows_cols; ++col) {
            const PEID pe = row * num_rows_cols + col;
            row_send_counts[row] += counts[pe];
        }
    }

    // --> Exchange counts
    std::vector<int> row_recv_counts(num_rows_cols);
    mpi::alltoall(row_send_counts.data(), 1, row_recv_counts.data(), 1, grid_comm.col_comm());

    // --> Compute displs
    std::vector<int> row_send_displs(num_rows_cols + 1);
    std::vector<int> row_recv_displs(num_rows_cols + 1);
    parallel::prefix_sum(row_send_counts.begin(), row_send_counts.end(), row_send_displs.begin() + 1);
    parallel::prefix_sum(row_recv_counts.begin(), row_recv_counts.end(), row_recv_displs.begin() + 1);

    //DBG << V(data) << V(counts) << V(row_send_counts) << V(row_recv_counts) << V(row_send_displs) << V(row_recv_displs);

    // --> Exchange payloads
    using MessageType = typename SendBuffer::value_type;
    NoinitVector<MessageType> row_data(row_recv_displs.back());
    mpi::alltoallv(
        data.data(), row_send_counts.data(), row_send_displs.data(), row_data.data(), row_recv_counts.data(),
        row_recv_displs.data(), grid_comm.col_comm());

    // --> Exchange counts within payload
    std::vector<int> row_counts(size);
    mpi::alltoall(counts.data(), num_rows_cols, row_counts.data(), num_rows_cols, grid_comm.col_comm());
    std::vector<int> row_displs(size + 1);
    parallel::prefix_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);

    //DBG << "After first hop: " << V(row_data) << V(row_counts) << V(row_displs);

    //
    // Step 2: Collect scattered data for each PE, and deliver the final payload
    //
    // Assertion: row_data contains data for PEs in the same row as the current PE:
    // col1, ..., coln, col1, ..., coln, col1, ..., coln
    // With sizes in row_counts:
    // size1, ..., sizen, size1, ..., sizen, size1, ..., sizen
    //
    std::vector<int> col_counts(num_rows_cols);
    std::vector<int> col_subcounts(size);

    for (PEID row = 0; row < num_rows_cols; ++row) {
        for (PEID col = 0; col < num_rows_cols; ++col) {
            const PEID pe = row * num_rows_cols + col;
            col_counts[col] += row_counts[pe];
            col_subcounts[row + col * num_rows_cols] = row_counts[pe];
        }
    }

    std::vector<int> col_displs(num_rows_cols + 1);
    parallel::prefix_sum(col_counts.begin(), col_counts.end(), col_displs.begin() + 1);

    NoinitVector<MessageType> col_data(row_data.size());
    tbb::parallel_for<PEID>(0, num_rows_cols, [&](const PEID col) {
        std::size_t i = col_displs[col];
        for (PEID row = 0; row < num_rows_cols; ++row) {
            const PEID pe        = row * num_rows_cols + col;
            const auto row_displ = row_displs[pe];
            const auto row_count = row_counts[pe];

            std::copy(row_data.begin() + row_displ, row_data.begin() + row_displ + row_count, col_data.begin() + i);
            i += row_count;
        }
    });

    // --> Exchange send counts within row
    std::vector<int> col_recv_counts(num_rows_cols);
    mpi::alltoall(col_counts.data(), 1, col_recv_counts.data(), 1, grid_comm.row_comm());

    std::vector<int> col_recv_displs(num_rows_cols + 1);
    parallel::prefix_sum(col_recv_counts.begin(), col_recv_counts.end(), col_recv_displs.begin() + 1);

    //DBG << "Before payload exchange: " << V(col_data) << V(col_counts) << V(col_displs) << V(col_recv_counts)
        //<< V(col_recv_displs) << V(col_subcounts);

    // --> Exchange payload
    NoinitVector<MessageType> final_data(col_recv_displs.back());
    mpi::alltoallv(
        col_data.data(), col_counts.data(), col_displs.data(), final_data.data(), col_recv_counts.data(),
        col_recv_displs.data(), grid_comm.row_comm());

    // --> Exchange counts within payload
    std::vector<int> final_subcounts(size);
    mpi::alltoall(col_subcounts.data(), num_rows_cols, final_subcounts.data(), num_rows_cols, grid_comm.row_comm());

    //DBG << "Final: " << V(final_data) << V(final_subcounts);

    std::size_t displ = 0;
    for (PEID col = 0; col < num_rows_cols; ++col) {
        for (PEID row = 0; row < num_rows_cols; ++row) {
            const PEID        index = col * num_rows_cols + row;
            const PEID        pe    = row * num_rows_cols + col;
            const std::size_t size  = final_subcounts[index];

            Buffer buffer(size);
            tbb::parallel_for<std::size_t>(
                0, final_subcounts[index], [&](const std::size_t i) { buffer[i] = final_data[displ + i]; });
            displ += size;

            invoke_receiver(std::move(buffer), pe, receiver);
        }
    }
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_grid(SendBuffers&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    // @todo avoid using this variant since it requires a full copy of the send buffers

    const auto [size, rank] = mpi::get_comm_info(comm);
    shm::NoinitVector<int> counts(size);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { counts[pe] = asserting_cast<int>(send_buffers[pe].size()); });
    shm::NoinitVector<int> displs(size + 1);
    shm::parallel::prefix_sum(counts.begin(), counts.end(), displs.begin() + 1);
    displs.front() = 0;

    shm::NoinitVector<Message> dense_buffer(displs.back());
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        std::copy(send_buffers[pe].begin(), send_buffers[pe].end(), dense_buffer.begin() + displs[pe]);
    });

    sparse_alltoall_grid<Message, Buffer>(std::move(dense_buffer), counts, std::forward<Receiver>(receiver), comm);
}
} // namespace dkaminpar::mpi
