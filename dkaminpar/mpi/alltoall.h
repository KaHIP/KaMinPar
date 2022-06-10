/*******************************************************************************
 * @file:   alltoall.h
 *
 * @author: Daniel Seemaier
 * @date:   10.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communications.
 ******************************************************************************/
#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "dkaminpar/mpi/wrapper.h"
#include "kaminpar/utils/timer.h"

#define SPARSE_ALLTOALL_NOFILTER \
    [](NodeID) {                 \
        return true;             \
    }

namespace dkaminpar::mpi {
template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(std::vector<Buffer>&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    SCOPED_TIMER("Sparse AllToAll", TIMER_DETAIL);

    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

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
        if (pe == rank) {
            if constexpr (receiver_invocable_with_pe) {
                receiver(std::move(send_buffers[rank]), pe);
            } else {
                receiver(std::move(send_buffers[rank]));
            }
        } else {
            auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, comm, MPI_STATUS_IGNORE);
            if constexpr (receiver_invocable_with_pe) {
                receiver(std::move(recv_buffer), pe);
            } else /* if (receiver_invocable_without_pe) */ {
                receiver(std::move(recv_buffer));
            }
        }
    }

    if (size > 1) {
        mpi::waitall(requests);
    }
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(const std::vector<Buffer>& send_buffers, Receiver&& receiver, MPI_Comm comm, const bool self) {
    SCOPED_TIMER("Sparse AllToAll", TIMER_DETAIL);

    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

    const auto [size, rank] = mpi::get_comm_info(comm);

    std::vector<MPI_Request> requests(size - 1 + self);

    std::size_t next_req_index = 0;
    for (PEID pe = 0; pe < size; ++pe) {
        if (self || pe != rank) {
            KASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
            KASSERT(next_req_index < requests.size());
            mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
        }
    }
    KASSERT(next_req_index == requests.size());

    for (PEID pe = 0; pe < size; ++pe) {
        if (self || pe != rank) {
            auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, comm, MPI_STATUS_IGNORE);
            if constexpr (receiver_invocable_with_pe) {
                receiver(std::move(recv_buffer), pe);
            } else /* if (receiver_invocable_without_pe) */ {
                receiver(std::move(recv_buffer));
            }
        }
    }

    mpi::waitall(requests);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer>&& send_buffers, MPI_Comm comm) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        std::move(send_buffers), [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); },
        comm);
    return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer>& send_buffers, MPI_Comm comm, const bool self) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        send_buffers, [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm, self);
    return recv_buffers;
}
} // namespace dkaminpar::mpi
