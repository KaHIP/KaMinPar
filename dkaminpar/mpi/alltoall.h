/*******************************************************************************
 * @file:   alltoall.h
 *
 * @author: Daniel Seemaier
 * @date:   10.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communications.
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "dkaminpar/mpi/wrapper.h"
#include "kaminpar/utils/timer.h"

#define SPARSE_ALLTOALL_NOFILTER \
    [](NodeID) {                 \
        return true;             \
    }

namespace dkaminpar::mpi {
namespace internal {
template <typename Buffer, typename Receiver>
void invoke_receiver(Buffer buffer, const PEID pe, const Receiver& receiver) {
    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

    if constexpr (receiver_invocable_with_pe) {
        receiver(std::move(buffer), pe);
    } else {
        receiver(std::move(buffer));
    }
}

template <typename SendBuffers, typename SendBuffer, typename Receiver>
void forward_self_buffer(SendBuffer& self_buffer, const PEID rank, const Receiver& receiver) {
    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, SendBuffer, PEID>;
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
void sparse_alltoall_complete(SendBuffers&& send_buffers, Receiver&& receiver, const bool self, MPI_Comm comm) {
    using namespace internal;

    const auto [size, rank] = mpi::get_comm_info(comm);

    std::vector<MPI_Request> requests(size - 1);
    std::size_t              next_req_index = 0;
    for (PEID pe = 0; pe < size; ++pe) {
        if (pe != rank) {
            KASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
            KASSERT(next_req_index < requests.size());
            mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
        }
    }
    KASSERT(next_req_index == requests.size());

    for (PEID pe = 0; pe < size; ++pe) {
        if (self && pe == rank) {
            forward_self_buffer<decltype(send_buffers)>(send_buffers[rank], rank, receiver);
        } else if (pe != rank) {
            auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, comm, MPI_STATUS_IGNORE);
            invoke_receiver(std::move(recv_buffer), pe, receiver);
        }
    }

    if (size > 1) {
        mpi::waitall(requests);
    }
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(const std::vector<Buffer>& send_buffers, Receiver&& receiver, const bool self, MPI_Comm comm) {
    SCOPED_TIMER("Sparse Alltoall", TIMER_DETAIL);
    sparse_alltoall_complete<Message, Buffer>(send_buffers, std::forward<Receiver>(receiver), self, comm);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(std::vector<Buffer>&& send_buffers, Receiver&& receiver, const bool self, MPI_Comm comm) {
    SCOPED_TIMER("Sparse Alltoall", TIMER_DETAIL);
    sparse_alltoall_complete<Message, Buffer>(std::move(send_buffers), std::forward<Receiver>(receiver), self, comm);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer>&& send_buffers, const bool self, MPI_Comm comm) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        std::move(send_buffers), [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); },
        self, comm);
    return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer>& send_buffers, MPI_Comm comm, const bool self) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        send_buffers, [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, self, comm);
    return recv_buffers;
}
} // namespace dkaminpar::mpi
