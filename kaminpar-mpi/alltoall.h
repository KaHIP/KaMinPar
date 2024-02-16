/*******************************************************************************
 * Implements sparse all-to-all communication.
 *
 * @file:   alltoall.h
 * @author: Daniel Seemaier
 * @date:   10.06.2022
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <mpi.h>
#include <tbb/parallel_for.h>

#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::mpi {
namespace internal {
template <typename Buffer, typename Receiver>
void invoke_receiver(Buffer buffer, const PEID pe, const Receiver &receiver) {
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  if constexpr (receiver_invocable_with_pe) {
    receiver(std::move(buffer), pe);
  } else {
    receiver(std::move(buffer));
  }
}

template <typename SendBuffers, typename SendBuffer, typename Receiver>
void forward_self_buffer(SendBuffer &self_buffer, const PEID rank, const Receiver &receiver) {
  constexpr bool receiver_invocable_with_pe =
      std::is_invocable_r_v<void, Receiver, SendBuffer, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, SendBuffer>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  if constexpr (std::is_lvalue_reference_v<SendBuffers>) {
    if constexpr (receiver_invocable_with_pe) {
      receiver(self_buffer, rank);
    } else {
      receiver(self_buffer);
    }
  } else {
    if constexpr (receiver_invocable_with_pe) {
      receiver(std::move(self_buffer), rank);
    } else {
      receiver(std::move(self_buffer));
    }
  }
}
} // namespace internal

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_alltoallv(SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm) {
  // Note: copies data twice which could be avoided
  using namespace internal;

  const auto [size, rank] = mpi::get_comm_info(comm);

  // START_TIMER("Alltoall construction");

  std::vector<int> send_counts(size);
  std::vector<int> recv_counts(size);
  std::vector<int> send_displs(size + 1);
  std::vector<int> recv_displs(size + 1);

  // Exchange send counts
  for (PEID pe = 0; pe < size; ++pe) {
    send_counts[pe] = asserting_cast<int>(send_buffers[pe].size());
  }
  parallel::prefix_sum(send_counts.begin(), send_counts.end(), send_displs.begin() + 1);
  mpi::alltoall(send_counts.data(), 1, recv_counts.data(), 1, comm);
  parallel::prefix_sum(recv_counts.begin(), recv_counts.end(), recv_displs.begin() + 1);

  // Build shared send buffer
  Buffer common_send_buffer;
  common_send_buffer.reserve(send_displs.back() + send_counts.back());
  for (PEID pe = 0; pe < size; ++pe) {
    for (const auto &e : send_buffers[pe]) {
      common_send_buffer.push_back(e);
    }

    if (!std::is_lvalue_reference_v<SendBuffers>) {
      // Free vector
      [[maybe_unused]] auto clear = std::move(send_buffers[pe]);
    }
  }

  // Exchange data
  Buffer common_recv_buffer(recv_displs.back() + recv_counts.back());

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI");

  mpi::alltoallv(
      common_send_buffer.data(),
      send_counts.data(),
      send_displs.data(),
      common_recv_buffer.data(),
      recv_counts.data(),
      recv_displs.data(),
      comm
  );

  // STOP_TIMER();
  // START_TIMER("Alltoall construction");

  // Call receiver
  std::vector<Buffer> recv_buffers(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    recv_buffers[pe].resize(recv_counts[pe]);
    tbb::parallel_for<int>(0, recv_counts[pe], [&](const int i) {
      recv_buffers[pe][i] = common_recv_buffer[recv_displs[pe] + i];
    });
  });

  // STOP_TIMER();
  // START_TIMER("Alltoall processing");

  for (PEID pe = 0; pe < size; ++pe) {
    invoke_receiver(std::move(recv_buffers[pe]), pe, receiver);
  }

  // STOP_TIMER();
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_complete(SendBuffers &&send_buffers, Receiver &&receiver, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);
  using namespace internal;

  // START_TIMER("Alltoall construction");

  std::vector<MPI_Request> requests(size - 1);
  std::size_t next_req_index = 0;
  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      KASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
      KASSERT(next_req_index < requests.size());
      mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
    }
  }
  KASSERT(next_req_index == requests.size());

  // STOP_TIMER();
  // START_TIMER("Alltoall MPI + processing");

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe == rank) {
      forward_self_buffer<decltype(send_buffers)>(send_buffers[rank], rank, receiver);
    } else if (pe != rank) {
      auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, comm, MPI_STATUS_IGNORE);
      invoke_receiver(std::move(recv_buffer), pe, receiver);
    }
  }

  if (size > 1) {
    mpi::waitall(requests);
  }

  // STOP_TIMER();
}
} // namespace kaminpar::mpi
