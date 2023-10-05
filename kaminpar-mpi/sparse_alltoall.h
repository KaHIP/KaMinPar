/*******************************************************************************
 * Facade for sparse alltoall operations.
 *
 * @file:   sparse_alltoall.h
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "kaminpar-mpi/alltoall.h"
#include "kaminpar-mpi/grid_alltoall.h"
#include "kaminpar-mpi/wrapper.h"

namespace kaminpar::mpi {
namespace tag {
struct complete_send_recv_tag {};
struct alltoallv_tag {};
struct grid_tag {};

constexpr static complete_send_recv_tag complete_send_recv;
constexpr static alltoallv_tag alltoallv;
constexpr static grid_tag grid;

// Used if no other implementation has priority
constexpr static auto default_sparse_alltoall = complete_send_recv;
} // namespace tag

constexpr static int SPARSE_GRID_ALLTOALL_THRESHOLD = 500;

/*
 * Implementation by tag
 */
template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall(
    tag::grid_tag, SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm
) {
  sparse_alltoall_grid<Message, Buffer>(
      std::forward<SendBuffers>(send_buffers), std::forward<Receiver>(receiver), comm
  );
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall(
    tag::alltoallv_tag, SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm
) {
  sparse_alltoall_alltoallv<Message, Buffer>(
      std::forward<SendBuffers>(send_buffers), std::forward<Receiver>(receiver), comm
  );
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall(
    tag::complete_send_recv_tag, SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm
) {
  sparse_alltoall_complete<Message, Buffer>(
      std::forward<SendBuffers>(send_buffers), std::forward<Receiver>(receiver), comm
  );
}

/*
 * Auto-dispatch implementation
 */

namespace internal {
template <typename SendBuffers>
bool use_sparse_grid_alltoall(const SendBuffers &send_buffers, MPI_Comm comm) {
  const std::size_t local_num_elements = parallel::accumulate(
      send_buffers.begin(),
      send_buffers.end(),
      0,
      [&](const auto &send_buffer) { return send_buffer.size(); }
  );

  const std::size_t global_num_elements = allreduce(local_num_elements, MPI_SUM, comm);

  const PEID size = get_comm_size(comm);
  return global_num_elements / size / size <= SPARSE_GRID_ALLTOALL_THRESHOLD;
}
}; // namespace internal

template <typename Message, typename Buffer = NoinitVector<Message>, typename Receiver>
void sparse_alltoall(const std::vector<Buffer> &send_buffers, Receiver &&receiver, MPI_Comm comm) {
  if (internal::use_sparse_grid_alltoall(send_buffers, comm)) {
    sparse_alltoall<Message, Buffer>(
        tag::grid, send_buffers, std::forward<Receiver>(receiver), comm
    );
  } else {
    sparse_alltoall<Message, Buffer>(
        tag::default_sparse_alltoall, send_buffers, std::forward<Receiver>(receiver), comm
    );
  }
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Receiver>
void sparse_alltoall(std::vector<Buffer> &&send_buffers, Receiver &&receiver, MPI_Comm comm) {
  if (internal::use_sparse_grid_alltoall(send_buffers, comm)) {
    sparse_alltoall<Message, Buffer>(
        tag::grid, std::move(send_buffers), std::forward<Receiver>(receiver), comm
    );
  } else {
    sparse_alltoall<Message, Buffer>(
        tag::default_sparse_alltoall,
        std::move(send_buffers),
        std::forward<Receiver>(receiver),
        comm
    );
  }
}

template <typename Message, typename Buffer = NoinitVector<Message>>
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer> &&send_buffers, MPI_Comm comm) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
  sparse_alltoall<Message, Buffer>(
      std::move(send_buffers),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); },
      comm
  );
  return recv_buffers;
}

template <typename Message, typename Buffer = NoinitVector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer> &send_buffers, MPI_Comm comm) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
  sparse_alltoall<Message, Buffer>(
      send_buffers,
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); },
      comm
  );
  return recv_buffers;
}

template <typename T>
void sparse_alltoallv(
    T *sendbuf,
    const int *sendcounts,
    const int *sdispls,
    T *recvbuf,
    const int *,
    const int *rdispls,
    MPI_Comm comm
) {
  const PEID size = mpi::get_comm_size(comm);
  std::vector<NoinitVector<T>> split_sendbuf;
  for (PEID pe = 0; pe < size; ++pe) {
    split_sendbuf.emplace_back(sendcounts[pe]);
    std::copy(
        sendbuf + sdispls[pe], sendbuf + sdispls[pe] + sendcounts[pe], split_sendbuf.back().begin()
    );
  }

  auto recv = sparse_alltoall_get<T, NoinitVector<T>>(split_sendbuf, comm);

  for (PEID pe = 0; pe < size; ++pe) {
    std::copy(recv[pe].begin(), recv[pe].end(), recvbuf + rdispls[pe]);
  }
}
} // namespace kaminpar::mpi
