/*******************************************************************************
 * @file:   graph_communication.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Collective MPI operations for graphs.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <omp.h>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/sparse_alltoall.h"
#include "dkaminpar/mpi/utils.h"

#include "common/assert.h"
#include "common/cache_aligned_vector.h"
#include "common/datastructures/marker.h"
#include "common/noinit_vector.h"
#include "common/parallel/aligned_element.h"
#include "common/timer.h"

#define SPARSE_ALLTOALL_NOFILTER \
    [](NodeID) {                 \
        return true;             \
    }

namespace kaminpar::mpi::graph {
using namespace kaminpar::dist;
SET_DEBUG(false);

template <typename Message, typename Buffer = NoinitVector<Message>, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_ghost(const DistributedGraph& graph, Builder&& builder, Receiver&& receiver) {
    sparse_alltoall_interface_to_ghost<Message, Buffer>(
        graph, SPARSE_ALLTOALL_NOFILTER, std::forward<Builder>(builder), std::forward<Receiver>(receiver)
    );
}

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_ghost(
    const DistributedGraph& graph, Filter&& filter, Builder&& builder, Receiver&& receiver
) {
    sparse_alltoall_interface_to_ghost<Message, Buffer>(
        graph, 0, graph.n(), std::forward<Filter>(filter), std::forward<Builder>(builder),
        std::forward<Receiver>(receiver)
    );
}

namespace internal {
template <typename Data>
void inclusive_col_prefix_sum(Data& data) {
    if (data.empty()) {
        return;
    }

    const std::size_t height = data.size();
    const std::size_t width  = data.front().size();

    for (std::size_t i = 1; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            data[i][j] += data[i - 1][j];
        }
    }
}
} // namespace internal

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_ghost(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, Builder&& builder,
    Receiver&& receiver
) {
    SCOPED_TIMER("Sparse AllToAll InterfaceToGhost");

    static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

    constexpr bool builder_invocable_with_pe    = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID, PEID>;
    constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID>;
    static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, const Buffer&, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, const Buffer&>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

    const auto [size, rank] = mpi::get_comm_info(graph.communicator());

    // allocate message counters
    const PEID                                     num_threads = omp_get_max_threads();
    std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

    // ASSERT that we count the same number of messages that we create
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    std::vector<parallel::Aligned<std::size_t>> total_num_messages(num_threads);
#endif

    // count messages to each PE for each thread
    START_TIMER("Count messages");
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    #pragma omp parallel for default(none) shared(from, to, graph, num_messages, filter, total_num_messages)
#else
    #pragma omp parallel for default(none) shared(from, to, graph, num_messages, filter)
#endif
    for (NodeID u = from; u < to; ++u) {
        if (!filter(u)) {
            continue;
        }

        const PEID thread = omp_get_thread_num();

        for (const auto [e, v]: graph.neighbors(u)) {
            if (graph.is_ghost_node(v)) {
                const PEID owner = graph.ghost_owner(v);
                ++num_messages[thread][owner];

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
                ++total_num_messages[thread];
#endif
            }
        }
    }

    // offset messages for each thread
    internal::inclusive_col_prefix_sum(num_messages);
    STOP_TIMER();

    // allocate send buffers
    START_TIMER("Allocation");
    std::vector<Buffer> send_buffers(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { send_buffers[pe].resize(num_messages.back()[pe]); });
    STOP_TIMER();

    // fill buffers
    START_TIMER("Partition messages");
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    #pragma omp parallel for default(none) \
        shared(send_buffers, from, to, filter, graph, builder, num_messages, total_num_messages)
#else
    #pragma omp parallel for default(none) shared(send_buffers, from, to, filter, graph, builder, num_messages)
#endif
    for (NodeID u = from; u < to; ++u) {
        if (!filter(u)) {
            continue;
        }

        const PEID thread = omp_get_thread_num();

        for (const auto [e, v]: graph.neighbors(u)) {
            if (graph.is_ghost_node(v)) {
                const PEID        pe   = graph.ghost_owner(v);
                const std::size_t slot = --num_messages[thread][pe];
                if constexpr (builder_invocable_with_pe) {
                    send_buffers[pe][slot] = builder(u, e, v, pe);
                } else /* if (builder_invocable_without_pe) */ {
                    send_buffers[pe][slot] = builder(u, e, v);
                }

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
                --total_num_messages[thread];
#endif
            }
        }
    }
    STOP_TIMER();

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    KASSERT(std::all_of(total_num_messages.begin(), total_num_messages.end(), [&](const auto& num_messages) {
        return num_messages == 0;
    }));
#endif

    sparse_alltoall<Message, Buffer>(
        std::move(send_buffers), std::forward<decltype(receiver)>(receiver), graph.communicator()
    );
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_ghost_get(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, Builder&& builder
) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
    sparse_alltoall_interface_to_ghost<Message, Buffer>(
        graph, from, to, std::forward<Filter>(filter), std::forward<Builder>(builder),
        [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
    );
    return recv_buffers;
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_pe(const DistributedGraph& graph, Builder&& builder, Receiver&& receiver) {
    sparse_alltoall_interface_to_pe<Message, Buffer>(
        graph, SPARSE_ALLTOALL_NOFILTER, std::forward<Builder>(builder), std::forward<Receiver>(receiver)
    );
}

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_pe(
    const DistributedGraph& graph, Filter&& filter, Builder&& builder, Receiver&& receiver
) {
    sparse_alltoall_interface_to_pe<Message, Buffer>(
        graph, 0, graph.n(), std::forward<Filter>(filter), std::forward<Builder>(builder),
        std::forward<Receiver>(receiver)
    );
}

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder, typename Receiver>
void sparse_alltoall_interface_to_pe(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, Builder&& builder,
    Receiver&& receiver
) {
    SCOPED_TIMER("Sparse AllToAll InterfaceToPE");

    static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

    constexpr bool builder_invocable_with_pe    = std::is_invocable_r_v<Message, Builder, NodeID, PEID>;
    constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID>;
    static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

    const PEID size = mpi::get_comm_size(graph.communicator());

    // allocate message counters
    const PEID                                     num_threads = omp_get_max_threads();
    std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

    // ASSERT that we count the same number of messages that we create
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    std::vector<parallel::Aligned<std::size_t>> total_num_messages(num_threads);
#endif

    // count messages to each PE for each thread
    START_TIMER("Count messages");
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    #pragma omp parallel default(none) shared(size, from, to, filter, graph, num_messages, total_num_messages)
#else
    #pragma omp parallel default(none) shared(size, from, to, filter, graph, num_messages)
#endif
    {
        Marker<>   created_message_for_pe(static_cast<std::size_t>(size));
        const PEID thread = omp_get_thread_num();

#pragma omp for
        for (NodeID u = from; u < to; ++u) {
            if (!filter(u)) {
                continue;
            }

            for (const auto [e, v]: graph.neighbors(u)) {
                if (!graph.is_ghost_node(v)) {
                    continue;
                }

                const PEID pe = graph.ghost_owner(v);

                if (created_message_for_pe.get(pe)) {
                    continue;
                }
                created_message_for_pe.set(pe);

                ++num_messages[thread][pe];

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
                ++total_num_messages[thread];
#endif
            }

            created_message_for_pe.reset();
        }
    }

    // offset messages for each thread
    internal::inclusive_col_prefix_sum(num_messages);
    STOP_TIMER();

    // allocate send buffers
    START_TIMER("Allocation");
    std::vector<Buffer> send_buffers(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { send_buffers[pe].resize(num_messages.back()[pe]); });
    STOP_TIMER();

    // fill buffers
    START_TIMER("Partition messages");
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    #pragma omp parallel default(none) \
        shared(send_buffers, size, from, to, builder, filter, graph, num_messages, total_num_messages)
#else
    #pragma omp parallel default(none) shared(send_buffers, size, from, to, builder, filter, graph, num_messages)
#endif
    {
        Marker<>   created_message_for_pe(static_cast<std::size_t>(size));
        const PEID thread = omp_get_thread_num();

#pragma omp for
        for (NodeID u = from; u < to; ++u) {
            if (!filter(u)) {
                continue;
            }

            for (const NodeID v: graph.adjacent_nodes(u)) {
                if (!graph.is_ghost_node(v)) {
                    continue;
                }

                const PEID pe = graph.ghost_owner(v);

                if (created_message_for_pe.get(pe)) {
                    continue;
                }
                created_message_for_pe.set(pe);

                const auto slot = --num_messages[thread][pe];

                if constexpr (builder_invocable_with_pe) {
                    send_buffers[pe][slot] = builder(u, pe);
                } else {
                    send_buffers[pe][slot] = builder(u);
                }

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
                --total_num_messages[thread];
#endif
            }

            created_message_for_pe.reset();
        }
    }
    STOP_TIMER();

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    KASSERT(std::all_of(total_num_messages.begin(), total_num_messages.end(), [&](const auto& num_messages) {
        return num_messages == 0;
    }));
#endif

    sparse_alltoall<Message, Buffer>(std::move(send_buffers), std::forward<Receiver>(receiver), graph.communicator());
} // namespace dkaminpar::mpi::graph

template <typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, Builder&& builder
) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
    sparse_alltoall_interface_to_pe<Message, Buffer>(
        graph, from, to, std::forward<Filter>(filter), std::forward<Builder>(builder),
        [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
    );
    return recv_buffers;
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename Builder>
std::vector<Buffer>
sparse_alltoall_interface_to_pe_get(const DistributedGraph& graph, Filter&& filter, Builder&& builder) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
    sparse_alltoall_interface_to_pe<Message, Buffer>(
        graph, 0, graph.n(), std::forward<Filter>(filter), std::forward<Builder>(builder),
        [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
    );
    return recv_buffers;
}

template <typename Message, typename Buffer = NoinitVector<Message>, typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(const DistributedGraph& graph, Builder&& builder) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
    sparse_alltoall_interface_to_pe<Message, Buffer>(
        graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER, std::forward<Builder>(builder),
        [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
    );
    return recv_buffers;
}

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename PEGetter, typename Builder,
    typename Receiver>
void sparse_alltoall_custom(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, PEGetter&& pe_getter,
    Builder&& builder, Receiver&& receiver
) {
    static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");
    static_assert(std::is_invocable_r_v<Message, Builder, NodeID>, "bad builder type");
    static_assert(std::is_invocable_r_v<PEID, PEGetter, NodeID>, "bad pe getter type");

    PEID size, rank;
    std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

    // allocate message counters
    const PEID                                     num_threads = omp_get_max_threads();
    std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

    // count messages to each PE for each thread
    START_TIMER("Count messages");
#pragma omp parallel default(none) shared(pe_getter, size, from, to, filter, graph, num_messages)
    {
        const PEID thread = omp_get_thread_num();
#pragma omp for
        for (NodeID u = from; u < to; ++u) {
            if (filter(u)) {
                ++num_messages[thread][pe_getter(u)];
            }
        }
    }

    // offset messages for each thread
    internal::inclusive_col_prefix_sum(num_messages);
    STOP_TIMER();

    // allocate send buffers
    START_TIMER("Allocation");
    std::vector<Buffer> send_buffers(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { send_buffers[pe].resize(num_messages.back()[pe]); });
    STOP_TIMER();

    // fill buffers
    START_TIMER("Partition messages");
#pragma omp parallel default(none) shared(pe_getter, send_buffers, size, from, to, builder, filter, graph, num_messages)
    {
        const PEID thread = omp_get_thread_num();
#pragma omp for
        for (NodeID u = from; u < to; ++u) {
            if (filter(u)) {
                const PEID pe          = pe_getter(u);
                const auto slot        = --num_messages[thread][pe];
                send_buffers[pe][slot] = builder(u);
            }
        }
    }
    STOP_TIMER();

    sparse_alltoall<Message, Buffer>(std::move(send_buffers), std::forward<Receiver>(receiver), graph.communicator());
}

template <
    typename Message, typename Buffer = NoinitVector<Message>, typename Filter, typename PEGetter, typename Builder>
std::vector<Buffer> sparse_alltoall_custom(
    const DistributedGraph& graph, const NodeID from, const NodeID to, Filter&& filter, PEGetter&& pe_getter,
    Builder&& builder
) {
    auto                size = mpi::get_comm_size(graph.communicator());
    std::vector<Buffer> recv_buffers(size);
    sparse_alltoall_custom<Message, Buffer>(
        graph, from, to, std::forward<Filter>(filter), std::forward<PEGetter>(pe_getter),
        std::forward<Builder>(builder),
        [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
    );
    return recv_buffers;
}
} // namespace kaminpar::mpi::graph
