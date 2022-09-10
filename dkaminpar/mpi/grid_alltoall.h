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
#include "dkaminpar/mpi/grid_topology.h"
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
    GridCommunicator(MPI_Comm comm) {
        SET_DEBUG(true);

        const auto [size, rank] = get_comm_info(comm);
        GridTopology topo(size);
        DBG << rank << " --> row=" << topo.row(rank) << " col=" << topo.virtual_col(rank);
        MPI_Comm_split(comm, topo.row(rank), rank, &_row_comm);
        MPI_Comm_split(comm, topo.virtual_col(rank), topo.virtual_col(rank) == topo.col(rank) ? rank : size + rank, &_col_comm);
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
    SET_DEBUG(true);

    using namespace internal;

    static GridCommunicator grid_comm(comm);

    const auto& row_comm      = grid_comm.row_comm();
    const PEID  row_comm_size = grid_comm.row_comm_size();
    const auto& col_comm      = grid_comm.col_comm();
    const PEID  col_comm_size = grid_comm.col_comm_size();

    const PEID size      = mpi::get_comm_size(comm);
    const PEID rank      = mpi::get_comm_rank(comm);
    const PEID grid_size = std::sqrt(size);

    GridTopology topo(size);
    const PEID   my_row         = topo.row(rank);
    const PEID   my_col         = topo.col(rank);
    const PEID   my_virtual_col = topo.virtual_col(rank);

    KASSERT(row_comm_size == topo.row_size(my_row));
    KASSERT(col_comm_size == topo.virtual_col_size(my_virtual_col), V(my_row) << V(my_col) << V(rank));

    DBG << V(my_row) << V(my_col) << V(my_virtual_col) << V(row_comm_size) << V(col_comm_size);

    static std::vector<int> row_counts_send_counts;
    static std::vector<int> row_counts_recv_counts;
    static std::vector<int> row_counts_send_displs;
    static std::vector<int> row_counts_recv_displs;
    if (row_counts_send_counts.empty()) {
        // Compute send counts
        DBG << "a";
        row_counts_send_counts.resize(col_comm_size);
        for (PEID row = 0; row < topo.num_rows(); ++row) {
            row_counts_send_counts[row] = topo.num_cols_in_row(row);
        }

        DBG << "b";

        // Compute recv counts
        row_counts_recv_counts.resize(col_comm_size);
        if (my_col == my_virtual_col) {
            std::fill(row_counts_recv_counts.begin(), row_counts_recv_counts.end(), topo.row_size(my_row));
        } else {
            std::fill(row_counts_recv_counts.begin(), row_counts_recv_counts.end(), 0);
        }

        DBG << "c";

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

    DBG << "alive 0";

    // Exchange counts within payload
    std::vector<int> row_counts(topo.row_size(my_row) * col_comm_size);

    DBG << V(rank) << V(row_counts_send_counts) << V(row_counts_send_displs) << V(row_counts_recv_counts) << V(row_counts_recv_displs);

    KASSERT(row_counts.size() >= row_counts_recv_displs.back());
    KASSERT(row_counts_send_counts.size() >= col_comm_size);
    KASSERT(counts.size() >= row_counts_send_displs.back());
    KASSERT(row_counts_recv_counts.size() >= col_comm_size);
    KASSERT(row_counts_recv_displs.size() >= col_comm_size);
    KASSERT(row_counts_send_displs.size() >= col_comm_size);

    mpi::alltoallv(
        counts.data(), row_counts_send_counts.data(), row_counts_send_displs.data(), row_counts.data(),
        row_counts_recv_counts.data(), row_counts_recv_displs.data(), col_comm
    );

    DBG << "alive 1 " << rank;

    // 1st hop send/recv counts/displs
    std::vector<int> row_send_counts(col_comm_size);
    for (PEID pe = 0; pe < size; ++pe) {
        const PEID row = topo.row(pe);
        row_send_counts[row] += counts[pe];
    }

    DBG << "alive 2";

    std::vector<int> row_recv_counts(col_comm_size);
    for (PEID row = 0; row < col_comm_size; ++row) {
        for (PEID col = 0; col < topo.row_size(my_row); ++col) {
            const PEID pe = row * col_comm_size + col;
            row_recv_counts[row] += row_counts[pe];
        }
    }

    DBG << "alive 3";

    std::vector<int> row_send_displs(grid_size + 1);
    std::vector<int> row_recv_displs(grid_size + 1);
    std::partial_sum(row_send_counts.begin(), row_send_counts.end(), row_send_displs.begin() + 1);
    std::partial_sum(row_recv_counts.begin(), row_recv_counts.end(), row_recv_displs.begin() + 1);

    // Exchange 1st hop payload
    std::vector<Message> row_recv_buf(row_recv_displs.back());

    KASSERT(row_send_counts.size() >= col_comm_size);
    KASSERT(row_send_displs.size() >= col_comm_size);
    KASSERT(data.size() >= row_send_displs.back());
    KASSERT(row_recv_counts.size() >= col_comm_size);
    KASSERT(row_recv_displs.size() >= col_comm_size);
    KASSERT(row_recv_buf.size() >= row_recv_displs.back());

    mpi::alltoallv(
        data.data(), row_send_counts.data(), row_send_displs.data(), row_recv_buf.data(), row_recv_counts.data(),
        row_recv_displs.data(), col_comm
    );

    std::vector<int> row_displs(size + 1);
    std::partial_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);

    DBG << V(row_recv_buf) << V(row_counts);

    // Assertion:
    // row_recv_buf containts data for each PE in the same row as this PE:
    // col1, ..., coln, col1, ..., coln, ... -- size: col size * row size
    // The sizes are given by row_counts:
    // size1, ..., sizen, size1, ..., sizen, ... -- size: col size * row size
    // The displacements are given by row_displs, thus, the data for PE 1 (in
    // row_comm) is given by row_data[displs(1) ... displs(1) + size(1)] AND
    // row_data[displs(n+1) ... displs(n+1) + size(n+1)] AND ...

    // total data count for each PE in my row
    std::vector<int> col_counts(row_comm_size);
    // for each PE in my row: col_comm_size * data count, i.e., for each PE in my column
    std::vector<int> col_subcounts(row_comm_size * col_comm_size);

    for (PEID col = 0; col < row_comm_size; ++col) {
        for (PEID from_row = 0; from_row < col_comm_size; ++from_row) {
            const PEID pe = col + from_row * row_comm_size;
            col_counts[col] += row_counts[pe];
            col_subcounts[from_row + col * col_comm_size] = row_counts[pe];
        }
    }

    static std::vector<int> col_subcounts_send_counts;
    static std::vector<int> col_subcounts_send_displs;
    static std::vector<int> col_subcounts_recv_counts;
    static std::vector<int> col_subcounts_recv_displs;
    if (col_subcounts_send_counts.empty()) {
        col_subcounts_send_counts.resize(row_comm_size);
        std::fill(col_subcounts_send_counts.begin(), col_subcounts_send_counts.end(), col_comm_size);

        col_subcounts_recv_counts.resize(row_comm_size);
        for (PEID col = 0; col < topo.num_full_cols(); ++col) {
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

    DBG << V(col_counts) << V(col_subcounts);
    DBG << V(col_subcounts_send_counts) << V(col_subcounts_send_displs) << V(col_subcounts_recv_counts)
        << V(col_subcounts_recv_displs);

    KASSERT(col_subcounts_recv_displs.back() == size);

    // Exchange counts
    std::vector<int> subcounts(size);

    KASSERT(col_subcounts_send_counts.size() >= row_comm_size);
    KASSERT(col_subcounts_recv_counts.size() >= row_comm_size);
    KASSERT(col_subcounts.size() >= col_subcounts_send_displs.back());
    KASSERT(col_subcounts_recv_counts.size() >= row_comm_size);
    KASSERT(col_subcounts_recv_displs.size() >= row_comm_size);
    KASSERT(subcounts.size() >= col_subcounts_recv_displs.back());

    mpi::alltoallv(
        col_subcounts.data(), col_subcounts_send_counts.data(), col_subcounts_send_displs.data(), subcounts.data(),
        col_subcounts_recv_counts.data(), col_subcounts_recv_displs.data(), row_comm
    );

    DBG << V(subcounts);

    // Assertion:
    // subcounts

    std::vector<int> col_recv_counts(row_comm_size);
    PEID _p = 0;
    for (PEID row = 0; row < topo.num_rows(); ++row) {
        int sum = 0;
        for (PEID col = 0; col < topo.num_cols_in_row(row); ++col) {
            sum += subcounts[_p++];
        }
        col_recv_counts[row] = sum;
    }

    // Transpose data buffer
    std::vector<int> col_recv_displs(row_comm_size + 1);
    std::partial_sum(col_recv_counts.begin(), col_recv_counts.end(), col_recv_displs.begin() + 1);
    std::vector<int> col_displs(row_comm_size + 1);
    std::partial_sum(col_counts.begin(), col_counts.end(), col_displs.begin() + 1);

    std::vector<Message> col_data(row_recv_buf.size());
    for (PEID col = 0; col < row_comm_size; ++col) {
        int i = col_displs[col];
        for (PEID row = 0; row < col_comm_size; ++row) {
            const PEID pe        = row * row_comm_size + col;
            const int  row_displ = row_displs[pe];
            const int  row_count = row_counts[pe];

            std::copy(
                row_recv_buf.begin() + row_displ, row_recv_buf.begin() + row_displ + row_count, col_data.begin() + i
            );
            i += row_count;
        }
    }

    // Exchange col payload
    DBG << V(col_recv_displs);
    std::vector<Message> col_recv_buf(col_recv_displs.back());

    DBG << V(col_data) << V(col_counts) << V(col_displs) << V(col_recv_counts) << V(col_recv_displs);

    KASSERT(col_counts.size() >= row_comm_size);
    KASSERT(col_displs.size() >= row_comm_size);
    KASSERT(col_data.size() >= col_displs.back());
    KASSERT(col_recv_counts.size() >= row_comm_size);
    KASSERT(col_recv_displs.size() >= row_comm_size);
    KASSERT(col_recv_buf.size() >= col_recv_displs.back());

    mpi::alltoallv(
        col_data.data(), col_counts.data(), col_displs.data(), col_recv_buf.data(), col_recv_counts.data(),
        col_recv_displs.data(), row_comm
    );

    // Assertion:
    // col_recv_buf contains all data adressed to this PE:
    // data from 0x0, 1x0, 2x0, ..., 0x1, 0x2, ... ...

    DBG << V(col_recv_buf) << V(col_recv_counts);

    std::size_t displ = 0;
    for (PEID pe = 0; pe < size; ++pe) {
        const PEID row         = topo.row(pe);
        const PEID virtual_col = topo.virtual_col(pe);

        const PEID index = row * topo.num_full_cols() + virtual_col;
        const auto size  = subcounts[index];

        Buffer buffer(size);
        tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) { buffer[i] = col_recv_buf[displ + i]; });
        displ += size;

        invoke_receiver(std::move(buffer), pe, receiver);
    }
    return;

    for (PEID col = 0; col < row_comm_size; ++col) {
        for (PEID row = 0; row < col_comm_size; ++row) {
            const PEID index = col * row_comm_size + row;
            const PEID pe    = row * col_comm_size + col;
            const auto size  = subcounts[index];

            Buffer buffer(size);
            tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) { buffer[i] = col_recv_buf[displ + i]; });
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
