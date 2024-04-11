/*******************************************************************************
 * Sparse all-to-all based on a 2D communication grid.
 *
 * @file:   grid_alltoall.h
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <numeric>

#include <mpi.h>
#include <tbb/parallel_for.h>

#include "kaminpar-mpi/alltoall.h"
#include "kaminpar-mpi/grid_topology.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/datastructures/preallocated_vector.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::mpi {
template <
    typename Message,
    typename Buffer,
    typename SendBuffer,
    typename CountsBuffer,
    typename Receiver>
void sparse_alltoall_grid(
    SendBuffer &&data, const CountsBuffer &counts, Receiver &&receiver, MPI_Comm comm
) {
  // START_TIMER("Alltoall construction");

  const GridCommunicator &grid_comm = get_grid_communicator(comm);
  const MPI_Comm &row_comm = grid_comm.row_comm();
  const MPI_Comm &col_comm = grid_comm.col_comm();

  const PEID row_comm_size = grid_comm.row_comm_size();
  const PEID col_comm_size = grid_comm.col_comm_size();

  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  const GridTopology topo(size);
  const PEID my_row = topo.row(rank);
  const PEID my_col = topo.col(rank);
  const PEID my_virtual_col = topo.virtual_col(rank);

  KASSERT(row_comm_size == topo.row_size(my_row));
  KASSERT(col_comm_size == topo.virtual_col_size(my_virtual_col));

  /*
   * Step 1 (column)
   *
   * For each row, send counts for all PEs in that row to the PE in our column
   * in that row
   */

  std::vector<int> row_counts_send_counts;
  std::vector<int> row_counts_recv_counts;
  std::vector<int> row_counts_send_displs;
  std::vector<int> row_counts_recv_displs;

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

  // Compute buffer displacements
  row_counts_send_displs.resize(col_comm_size + 1);
  row_counts_recv_displs.resize(col_comm_size + 1);
  std::partial_sum(
      row_counts_send_counts.begin(),
      row_counts_send_counts.end(),
      row_counts_send_displs.begin() + 1
  );
  std::partial_sum(
      row_counts_recv_counts.begin(),
      row_counts_recv_counts.end(),
      row_counts_recv_displs.begin() + 1
  );

  std::vector<int> row_counts(row_comm_size * col_comm_size);

  // Exchange counts within payload
  KASSERT(asserting_cast<int>(row_counts.size()) >= row_counts_recv_displs.back());
  KASSERT(asserting_cast<int>(row_counts_send_counts.size()) >= col_comm_size);
  KASSERT(asserting_cast<int>(counts.size()) >= row_counts_send_displs.back());
  KASSERT(asserting_cast<int>(row_counts_recv_counts.size()) >= col_comm_size);
  KASSERT(asserting_cast<int>(row_counts_recv_displs.size()) >= col_comm_size);
  KASSERT(asserting_cast<int>(row_counts_send_displs.size()) >= col_comm_size);

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI");

  mpi::alltoallv(
      counts.data(),
      row_counts_send_counts.data(),
      row_counts_send_displs.data(),
      row_counts.data(),
      row_counts_recv_counts.data(),
      row_counts_recv_displs.data(),
      col_comm
  );

  // STOP_TIMER();
  // START_TIMER("Alltoall construction");

  /*
   * After Step 1 (column)
   *
   * row_counts contains data counts for PEs in my row:
   * from PE in row 0: data count for col0, col1, col2, ...,  (row_comm_size
   * items) from PE in row 1: data count for col0, col1, col2, ...,
   * (row_comm_size items)
   * ...
   *
   * Size: row_comm_size * col_comm_size
   */

  /*
   * Step 2 (column)
   *
   * Exchange data within our column: send PE in row 0 all data for row 0, ...
   */

  std::vector<int> row_send_counts(col_comm_size);
  std::vector<int> row_recv_counts(col_comm_size);
  std::vector<int> row_send_displs(col_comm_size + 1);
  std::vector<int> row_recv_displs(col_comm_size + 1);
  std::vector<int> row_displs(row_comm_size * col_comm_size + 1);

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

  std::vector<Message> row_recv_buf(row_recv_displs.back());

  KASSERT(asserting_cast<PEID>(row_send_counts.size()) >= col_comm_size);
  KASSERT(asserting_cast<PEID>(row_send_displs.size()) >= col_comm_size);
  KASSERT(asserting_cast<int>(data.size()) >= row_send_displs.back());
  KASSERT(asserting_cast<PEID>(row_recv_counts.size()) >= col_comm_size);
  KASSERT(asserting_cast<PEID>(row_recv_displs.size()) >= col_comm_size);
  KASSERT(asserting_cast<int>(row_recv_buf.size()) >= row_recv_displs.back());

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI");

  mpi::alltoallv(
      data.data(),
      row_send_counts.data(),
      row_send_displs.data(),
      row_recv_buf.data(),
      row_recv_counts.data(),
      row_recv_displs.data(),
      col_comm
  );

  // STOP_TIMER();
  // START_TIMER("Alltoall construction");

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

  std::vector<int> col_counts(row_comm_size);
  std::vector<int> col_subcounts(row_comm_size * col_comm_size);
  std::vector<int> subcounts(size);
  std::vector<int> col_recv_counts(row_comm_size);
  std::vector<int> col_recv_displs(row_comm_size + 1);
  std::vector<int> col_displs(row_comm_size + 1);

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

  std::vector<int> col_subcounts_send_counts;
  std::vector<int> col_subcounts_send_displs;
  std::vector<int> col_subcounts_recv_counts;
  std::vector<int> col_subcounts_recv_displs;
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
      col_subcounts_send_counts.begin(),
      col_subcounts_send_counts.end(),
      col_subcounts_send_displs.begin() + 1
  );
  std::partial_sum(
      col_subcounts_recv_counts.begin(),
      col_subcounts_recv_counts.end(),
      col_subcounts_recv_displs.begin() + 1
  );

  // Exchange counts
  KASSERT(asserting_cast<PEID>(col_subcounts_send_counts.size()) >= row_comm_size);
  KASSERT(asserting_cast<PEID>(col_subcounts_recv_counts.size()) >= row_comm_size);
  KASSERT(asserting_cast<int>(col_subcounts.size()) >= col_subcounts_send_displs.back());
  KASSERT(asserting_cast<PEID>(col_subcounts_recv_counts.size()) >= row_comm_size);
  KASSERT(asserting_cast<PEID>(col_subcounts_recv_displs.size()) >= row_comm_size);
  KASSERT(asserting_cast<int>(subcounts.size()) >= col_subcounts_recv_displs.back());

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI");

  mpi::alltoallv(
      col_subcounts.data(),
      col_subcounts_send_counts.data(),
      col_subcounts_send_displs.data(),
      subcounts.data(),
      col_subcounts_recv_counts.data(),
      col_subcounts_recv_displs.data(),
      row_comm
  );

  // STOP_TIMER();
  // START_TIMER("Alltoall construction");

  for (PEID pe = 0, row = 0; row < topo.num_full_cols(); ++row) {
    int sum = 0;

    for (PEID col = 0; col < topo.virtual_col_size(row); ++col) {
      KASSERT(pe < asserting_cast<PEID>(subcounts.size()));
      sum += subcounts[pe++];
    }

    KASSERT(row < asserting_cast<PEID>(col_recv_counts.size()));
    col_recv_counts[row] = sum;
  }

  // Transpose data buffer
  std::vector<Message> col_data(row_recv_buf.size());

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
          row_recv_buf.begin() + row_displ,
          row_recv_buf.begin() + row_displ + row_count,
          col_data.begin() + i
      );
      i += row_count;
    }
  }

  // Exchange col payload
  std::vector<Message> col_recv_buf(col_recv_displs.back());

  KASSERT(asserting_cast<PEID>(col_counts.size()) >= row_comm_size);
  KASSERT(asserting_cast<PEID>(col_displs.size()) >= row_comm_size);
  KASSERT(asserting_cast<int>(col_data.size()) >= col_displs.back());
  KASSERT(asserting_cast<PEID>(col_recv_counts.size()) >= row_comm_size);
  KASSERT(asserting_cast<PEID>(col_recv_displs.size()) >= row_comm_size);
  KASSERT(asserting_cast<int>(col_recv_buf.size()) >= col_recv_displs.back());

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI");

  mpi::alltoallv(
      col_data.data(),
      col_counts.data(),
      col_displs.data(),
      col_recv_buf.data(),
      col_recv_counts.data(),
      col_recv_displs.data(),
      row_comm
  );

  // STOP_TIMER();

  // Assertion:
  // col_recv_buf contains all data adressed to this PE:
  // data from 0x0, 1x0, 2x0, ..., 0x1, 0x2, ... ...

  // START_TIMER("Alltoall processing");

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
      internal::invoke_receiver(std::move(buffer), pe, receiver);
    }
  }

  // STOP_TIMER();
}

// @todo avoid using this variant since it requires a full copy of the send
// buffers
template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_grid(SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm) {
  // START_TIMER("Alltoall construction");

  const auto [size, rank] = mpi::get_comm_info(comm);
  NoinitVector<int> counts(size);

  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    counts[pe] = asserting_cast<int>(send_buffers[pe].size());
  });
  NoinitVector<int> displs(size + 1);
  parallel::prefix_sum(counts.begin(), counts.end(), displs.begin() + 1);
  displs.front() = 0;

  NoinitVector<Message> dense_buffer(displs.back());
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    std::copy(send_buffers[pe].begin(), send_buffers[pe].end(), dense_buffer.begin() + displs[pe]);
  });

  // STOP_TIMER();

  sparse_alltoall_grid<Message, Buffer>(
      std::move(dense_buffer), counts, std::forward<Receiver>(receiver), comm
  );
}
} // namespace kaminpar::mpi
