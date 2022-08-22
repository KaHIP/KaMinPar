/*******************************************************************************
 * @file:   alltoall.h
 * @author: Daniel Seemaier
 * @date:   10.06.2022
 * @brief:  Algorithms to perform (sparse) all-to-all communications.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/noinit_vector.h"
#include "common/timer.h"

#define SPARSE_ALLTOALL_NOFILTER \
    [](NodeID) {                 \
        return true;             \
    }

namespace kaminpar::mpi {
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
void sparse_alltoall_sparse(SendBuffers&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    using namespace internal;

    thread_local static int tag_counter = 0;
    const int               tag         = tag_counter++;

    const auto [size, rank] = mpi::get_comm_info(comm);

    std::vector<MPI_Request> requests;
    requests.reserve(size);
    std::vector<std::uint8_t> sends_message_to(size);

    // Send MPI messages
    for (PEID pe = 0; pe < size; ++pe) {
        if (pe == rank || send_buffers[pe].empty()) {
            continue;
        }

        sends_message_to[pe] = true;
        requests.emplace_back();

        MPI_Issend(
            send_buffers[pe].data(), static_cast<int>(send_buffers[pe].size()), mpi::type::get<Message>(), pe, tag,
            comm, &requests.back()
        );
    }

    if (!send_buffers[rank].empty()) {
        forward_self_buffer<decltype(send_buffers)>(send_buffers[rank], rank, receiver);
    }

    // Receive messages until MPI_Issend is completed
    int isend_done = 0;
    while (isend_done == 0) {
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;

            MPI_Status status{};
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, mpi::type::get<Message>(), &count);
                Buffer recv_buffer(count);
                mpi::recv(recv_buffer.data(), count, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);

                invoke_receiver(std::move(recv_buffer), status.MPI_SOURCE, receiver);
            }
        }

        isend_done = 0;
        MPI_Testall(asserting_cast<int>(requests.size()), requests.data(), &isend_done, MPI_STATUSES_IGNORE);
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(comm, &barrier_request);

    // Receive messages until all PEs reached the barrier
    int ibarrier_done = 0;
    while (ibarrier_done == 0) {
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;

            MPI_Status status{};
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, mpi::type::get<Message>(), &count);
                Buffer recv_buffer(count);
                mpi::recv(recv_buffer.data(), count, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);

                invoke_receiver(std::move(recv_buffer), status.MPI_SOURCE, receiver);
            }
        }

        // Test if all PEs reached the Ibarrier
        MPI_Test(&barrier_request, &ibarrier_done, MPI_STATUS_IGNORE);
    }
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_alltoallv(SendBuffers&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    // Note: copies data twice which could be avoided

    const auto [size, rank] = mpi::get_comm_info(comm);
    using namespace internal;

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
        for (const auto& e: send_buffers[pe]) {
            common_send_buffer.push_back(e);
        }

        if (!std::is_lvalue_reference_v<SendBuffers>) {
            std::move(send_buffers[pe]); // clear
        }
    }

    // Exchange data
    Buffer common_recv_buffer(recv_displs.back() + recv_counts.back());
    START_TIMER("MPI_Alltoallv", TIMER_DETAIL);
    mpi::alltoallv(
        common_send_buffer.data(), send_counts.data(), send_displs.data(), common_recv_buffer.data(),
        recv_counts.data(), recv_displs.data(), comm
    );
    STOP_TIMER();

    // Call receiver
    std::vector<Buffer> recv_buffers(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        recv_buffers[pe].resize(recv_counts[pe]);
        tbb::parallel_for<int>(0, recv_counts[pe], [&](const int i) {
            recv_buffers[pe][i] = common_recv_buffer[recv_displs[pe] + i];
        });
    });

    for (PEID pe = 0; pe < size; ++pe) {
        invoke_receiver(std::move(recv_buffers[pe]), pe, receiver);
    }
}

template <typename Message, typename Buffer, typename SendBuffers, typename Receiver>
void sparse_alltoall_complete(SendBuffers&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    const auto [size, rank] = mpi::get_comm_info(comm);
    using namespace internal;

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

template <typename Message, typename Buffer = NoinitVector<Message>, typename Receiver>
void sparse_alltoall(const std::vector<Buffer>& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    SCOPED_TIMER("Sparse Alltoall", TIMER_DETAIL);
    sparse_alltoall_complete<Message, Buffer>(send_buffers, std::forward<Receiver>(receiver), comm);
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Receiver>
void sparse_alltoall(std::vector<Buffer>&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    SCOPED_TIMER("Sparse Alltoall", TIMER_DETAIL);
    sparse_alltoall_complete<Message, Buffer>(std::move(send_buffers), std::forward<Receiver>(receiver), comm);
}

template <typename Message, typename Buffer = NoinitVector<Message>>
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer>&& send_buffers, MPI_Comm comm) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        std::move(send_buffers), [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); },
        comm
    );
    return recv_buffers;
}

template <typename Message, typename Buffer = NoinitVector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer>& send_buffers, MPI_Comm comm) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        send_buffers, [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm
    );
    return recv_buffers;
}
} // namespace kaminpar::mpi
