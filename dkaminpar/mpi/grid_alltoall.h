/*******************************************************************************
 * @file:   grid_alltoall.h
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communication.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/preallocated_vector.h"
#include "common/timer.h"
#include "common/utils/math.h"

namespace kaminpar::mpi {
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

    const PEID              size      = mpi::get_comm_size(comm);
    const PEID              rank      = mpi::get_comm_rank(comm);
    const PEID              grid_size = std::sqrt(size);
    static GridCommunicator grid_comm(size, rank, comm);

    // Exchange counts within payload
    std::vector<int> row_counts(size);
    mpi::alltoall(counts.data(), grid_size, row_counts.data(), grid_size, grid_comm.col_comm());

    // 1st hop send/recv counts/displs
    std::vector<int> row_send_counts(grid_size);
    for (PEID row = 0; row < grid_size; ++row) {
        for (PEID col = 0; col < grid_size; ++col) {
            const PEID pe = row * grid_size + col;
            row_send_counts[row] += counts[pe];
        }
    }

    std::vector<int> row_recv_counts(grid_size);
    for (PEID row = 0; row < grid_size; ++row) {
        for (PEID col = 0; col < grid_size; ++col) {
            const PEID pe = row * grid_size + col;
            row_recv_counts[row] += row_counts[pe];
        }
    }

    std::vector<int> row_send_displs(grid_size + 1);
    std::vector<int> row_recv_displs(grid_size + 1);
    std::partial_sum(row_send_counts.begin(), row_send_counts.end(), row_send_displs.begin() + 1);
    std::partial_sum(row_recv_counts.begin(), row_recv_counts.end(), row_recv_displs.begin() + 1);

    // Exchange 1st hop payload
    std::vector<Message> row_recv_buf(row_recv_displs.back());
    mpi::alltoallv(
        data.data(), row_send_counts.data(), row_send_displs.data(), row_recv_buf.data(), row_recv_counts.data(),
        row_recv_displs.data(), grid_comm.col_comm()
    );

    std::vector<int> row_displs(size + 1);
    std::partial_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);

    // Assertion:
    // row_data containts data for each PE in the same row as this PE:
    // col1, ..., coln, col1, ..., coln, ...
    // The sizes are given by row_counts:
    // size1, ..., sizen, size1, ..., sizen, ...
    // The displacements are given by row_displs, thus, the data for PE 1 (in
    // row_comm) is given by row_data[displs(1) ... displs(1) + size(1)] AND
    // row_data[displs(n+1) ... displs(n+1) + size(n+1)] AND ...

    std::vector<int> col_counts(grid_size);
    std::vector<int> col_subcounts(size);

    for (PEID row = 0; row < grid_size; ++row) {
        for (PEID col = 0; col < grid_size; ++col) {
            const PEID pe = row * grid_size + col;
            col_counts[col] += row_counts[pe];
            col_subcounts[row + col * grid_size] = row_counts[pe];
        }
    }

    std::vector<int> col_displs(grid_size + 1);
    std::partial_sum(col_counts.begin(), col_counts.end(), col_displs.begin() + 1);

    std::vector<Message> col_data(row_recv_buf.size());
    for (PEID col = 0; col < grid_size; ++col) {
        int i = col_displs[col];
        for (PEID row = 0; row < grid_size; ++row) {
            const PEID pe        = row * grid_size + col;
            const int row_displ = row_displs[pe];
            const int row_count = row_counts[pe];

            std::copy(
                row_recv_buf.begin() + row_displ, row_recv_buf.begin() + row_displ + row_count, col_data.begin() + i
            );
            i += row_count;
        }
    }

    // Exchange counts
    std::vector<int> subcounts(size);
    mpi::alltoall(col_subcounts.data(), grid_size, subcounts.data(), grid_size, grid_comm.row_comm());

    std::vector<int> col_recv_counts(grid_size);
    for (PEID row = 0; row < grid_size; ++row) {
        int sum = 0;
        for (PEID col = 0; col < grid_size; ++col) {
            sum += subcounts[row * grid_size + col];
        }
        col_recv_counts[row] = sum;
    }

    std::vector<int> col_recv_displs(grid_size + 1);
    std::partial_sum(col_recv_counts.begin(), col_recv_counts.end(), col_recv_displs.begin() + 1);

    // Exchange col payload
    std::vector<Message> col_recv_buf(col_recv_displs.back());

    mpi::alltoallv(
        col_data.data(), col_counts.data(), col_displs.data(), col_recv_buf.data(), col_recv_counts.data(),
        col_recv_displs.data(), grid_comm.row_comm()
    );

    std::size_t displ = 0;
    for (PEID col = 0; col < grid_size; ++col) {
        for (PEID row = 0; row < grid_size; ++row) {
            const PEID index = col * grid_size + row;
            const PEID pe    = row * grid_size + col;
            const auto size  = subcounts[index];

            Buffer buffer(size);
            tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) { buffer[i] = col_recv_buf[displ + i]; });
            displ += size;

            invoke_receiver(std::move(buffer), pe, receiver);
        }
    }
}

template <typename Message, typename Buffer, typename SendBuffer, typename CountsBuffer, typename Receiver>
void sparse_alltoall_grid2(SendBuffer&& data, const CountsBuffer& counts, Receiver&& receiver, MPI_Comm comm) {
    using namespace internal;
    using namespace kaminpar;

    const auto [size, rank] = mpi::get_comm_info(comm);
    KASSERT(math::is_square(size), "", assert::always);
    static GridCommunicator grid_comm(size, rank, comm);
    const PEID              num_rows_cols = grid_comm.sqrt();

    SET_DEBUG(false);
    // DBG << V(num_rows_cols);

    //
    // Step 1: Send rows to the right PE in the same column
    //
    std::vector<int> row_counts(size);
    mpi::alltoall(counts.data(), num_rows_cols, row_counts.data(), num_rows_cols, MPI_INT, grid_comm.col_comm());

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
    for (PEID row = 0; row < num_rows_cols; ++row) {
        for (PEID col = 0; col < num_rows_cols; ++col) {
            const PEID pe = row * num_rows_cols + col;
            row_recv_counts[row] += row_counts[pe];
        }
    }

    // --> Compute displs
    std::vector<int> row_send_displs(num_rows_cols + 1);
    std::vector<int> row_recv_displs(num_rows_cols + 1);
    parallel::prefix_sum(row_send_counts.begin(), row_send_counts.end(), row_send_displs.begin() + 1);
    parallel::prefix_sum(row_recv_counts.begin(), row_recv_counts.end(), row_recv_displs.begin() + 1);

    // DBG << V(data) << V(counts) << V(row_send_counts) << V(row_recv_counts) << V(row_send_displs) <<
    // V(row_recv_displs);

    // --> Exchange payloads
    using MessageType = typename SendBuffer::value_type;
    NoinitVector<MessageType> row_data(row_recv_displs.back());
    mpi::alltoallv(
        data.data(), row_send_counts.data(), row_send_displs.data(), row_data.data(), row_recv_counts.data(),
        row_recv_displs.data(), grid_comm.col_comm()
    );

    // --> Exchange counts within payload
    std::vector<int> row_displs(size + 1);
    parallel::prefix_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);

    // DBG << "After first hop: " << V(row_data) << V(row_counts) << V(row_displs);

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
    std::vector<int> subcounts(size);
    std::vector<int> col_recv_counts(num_rows_cols);
    mpi::alltoall(col_counts.data(), 1, col_recv_counts.data(), 1, grid_comm.row_comm());

    std::vector<int> col_recv_displs(num_rows_cols + 1);
    parallel::prefix_sum(col_recv_counts.begin(), col_recv_counts.end(), col_recv_displs.begin() + 1);

    // DBG << "Before payload exchange: " << V(col_data) << V(col_counts) << V(col_displs) << V(col_recv_counts)
    //<< V(col_recv_displs) << V(col_subcounts);

    // --> Exchange payload
    NoinitVector<MessageType> final_data(col_recv_displs.back());
    mpi::alltoallv(
        col_data.data(), col_counts.data(), col_displs.data(), final_data.data(), col_recv_counts.data(),
        col_recv_displs.data(), grid_comm.row_comm()
    );

    // --> Exchange counts within payload
    std::vector<int> final_subcounts(size);
    mpi::alltoall(col_subcounts.data(), num_rows_cols, final_subcounts.data(), num_rows_cols, grid_comm.row_comm());

    // DBG << "Final: " << V(final_data) << V(final_subcounts);

    std::size_t displ = 0;
    for (PEID col = 0; col < num_rows_cols; ++col) {
        for (PEID row = 0; row < num_rows_cols; ++row) {
            const PEID        index = col * num_rows_cols + row;
            const PEID        pe    = row * num_rows_cols + col;
            const std::size_t size  = final_subcounts[index];

            Buffer buffer(size);
            tbb::parallel_for<std::size_t>(0, final_subcounts[index], [&](const std::size_t i) {
                buffer[i] = final_data[displ + i];
            });
            displ += size;

            invoke_receiver(std::move(buffer), pe, receiver);
        }
    }
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
