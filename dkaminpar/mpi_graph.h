/*******************************************************************************
 * @file:   mpi_graph.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Collective MPI operations for graphs.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/utility/timer.h"

#include <omp.h>
#include <tbb/concurrent_vector.h>
#include <type_traits>

namespace dkaminpar::mpi::graph {
SET_DEBUG(false);

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, SPARSE_ALLTOALL_NOFILTER,
                                                      std::forward<decltype(builder)>(builder),
                                                      std::forward<decltype(receiver)>(receiver));
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, auto &&filter, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, 0, graph.n(), std::forward<decltype(filter)>(filter),
                                                      std::forward<decltype(builder)>(builder),
                                                      std::forward<decltype(receiver)>(receiver));
}

namespace internal {
void inclusive_col_prefix_sum(auto &data) {
  if (data.empty()) {
    return;
  }

  const std::size_t height = data.size();
  const std::size_t width = data.front().size();

  for (std::size_t i = 1; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      data[i][j] += data[i - 1][j];
    }
  }
}
} // namespace internal

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, const NodeID from, const NodeID to,
                                        auto &&filter, auto &&builder, auto &&receiver) {
  SCOPED_TIMER("Sparse AllToAll InterfaceToGhost");

  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  constexpr bool builder_invocable_with_pe = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID, PEID>;
  constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID>;
  static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, const Buffer &, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, const Buffer &>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // allocate message counters
  const PEID num_threads = omp_get_max_threads();
  std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

  // count messages to each PE for each thread
  START_TIMER("Count messages", TIMER_FINE);
#pragma omp parallel for default(none) shared(from, to, graph, num_messages, filter)
  for (NodeID u = from; u < to; ++u) {
    if (!filter(u)) {
      continue;
    }

    const PEID thread = omp_get_thread_num();

    for (const auto [e, v] : graph.neighbors(u)) {
      if (graph.is_ghost_node(v)) {
        const PEID owner = graph.ghost_owner(v);
        ++num_messages[thread][owner];
      }
    }
  }

  // offset messages for each thread
  internal::inclusive_col_prefix_sum(num_messages);
  STOP_TIMER(TIMER_FINE);

  // allocate send buffers
  START_TIMER("Allocation", TIMER_FINE);
  std::vector<Buffer> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) {
    send_buffers.emplace_back(num_messages.back()[pe]);
  }
  STOP_TIMER(TIMER_FINE);

  // fill buffers
  START_TIMER("Partition messages", TIMER_FINE);
#pragma omp parallel for default(none) shared(send_buffers, from, to, filter, graph, builder, num_messages)
  for (NodeID u = from; u < to; ++u) {
    if (!filter(u)) {
      continue;
    }

    const PEID thread = omp_get_thread_num();

    for (const auto [e, v] : graph.neighbors(u)) {
      if (graph.is_ghost_node(v)) {
        const PEID pe = graph.ghost_owner(v);
        const std::size_t slot = --num_messages[thread][pe];
        if constexpr (builder_invocable_with_pe) {
          send_buffers[pe][slot] = builder(u, e, v, pe);
        } else /* if (builder_invocable_without_pe) */ {
          send_buffers[pe][slot] = builder(u, e, v);
        }
      }
    }
  }
  STOP_TIMER(TIMER_FINE);

  sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator());
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_interface_to_ghost_get(const DistributedGraph &graph, const NodeID from,
                                                           const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph, from, to, std::forward<decltype(filter)>(filter), std::forward<decltype(builder)>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); });
  return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, SPARSE_ALLTOALL_NOFILTER,
                                                   std::forward<decltype(builder)>(builder),
                                                   std::forward<decltype(receiver)>(receiver));
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, auto &&filter, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, 0, graph.n(), std::forward<decltype(filter)>(filter),
                                                   std::forward<decltype(builder)>(builder),
                                                   std::forward<decltype(receiver)>(receiver));
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, const NodeID from, const NodeID to, auto &&filter,
                                     auto &&builder, auto &&receiver) {
  SCOPED_TIMER("Sparse AllToAll InterfaceToPE");

  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  constexpr bool builder_invocable_with_pe = std::is_invocable_r_v<Message, Builder, NodeID, PEID>;
  constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID>;
  static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

  const PEID size = mpi::get_comm_size(graph.communicator());

  // allocate message counters
  const PEID num_threads = omp_get_max_threads();
  std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

  // count messages to each PE for each thread
  START_TIMER("Count messages", TIMER_FINE);
#pragma omp parallel default(none) shared(size, from, to, filter, graph, num_messages)
  {
    shm::Marker<> created_message_for_pe(static_cast<std::size_t>(size));
    const PEID thread = omp_get_thread_num();

#pragma omp for
    for (NodeID u = from; u < to; ++u) {
      if (!filter(u)) {
        continue;
      }

      for (const auto [e, v] : graph.neighbors(u)) {
        if (!graph.is_ghost_node(v)) {
          continue;
        }

        const PEID pe = graph.ghost_owner(v);

        if (created_message_for_pe.get(pe)) {
          continue;
        }
        created_message_for_pe.set(pe);

        ++num_messages[thread][pe];
      }

      created_message_for_pe.reset();
    }
  }

  // offset messages for each thread
  internal::inclusive_col_prefix_sum(num_messages);
  STOP_TIMER(TIMER_FINE);

  // allocate send buffers
  START_TIMER("Allocation", TIMER_FINE);
  std::vector<Buffer> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) {
    send_buffers.emplace_back(num_messages.back()[pe]);
  }
  STOP_TIMER(TIMER_FINE);

  // fill buffers
  START_TIMER("Partition messages", TIMER_FINE);
#pragma omp parallel default(none) shared(send_buffers, size, from, to, builder, filter, graph, num_messages)
  {
    shm::Marker<> created_message_for_pe(static_cast<std::size_t>(size));
    const PEID thread = omp_get_thread_num();

#pragma omp for
    for (NodeID u = from; u < to; ++u) {
      if (!filter(u)) {
        continue;
      }

      for (const NodeID v : graph.adjacent_nodes(u)) {
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
      }

      created_message_for_pe.reset();
    }
  }
  STOP_TIMER(TIMER_FINE);

  sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator());
} // namespace dkaminpar::mpi::graph

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(const DistributedGraph &graph, const NodeID from,
                                                        const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph, from, to, std::forward<decltype(filter)>(filter), std::forward<decltype(builder)>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); });
  return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(const DistributedGraph &graph, auto &&filter, auto &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph, 0, graph.n(), std::forward<decltype(filter)>(filter), std::forward<decltype(builder)>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); });
  return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(const DistributedGraph &graph, auto &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER, std::forward<decltype(builder)>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); });
  return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall_custom(const DistributedGraph &graph, const NodeID from, const NodeID to, auto &&filter,
                            auto &&pe_getter, auto &&builder, auto &&receiver, const bool self = false) {
  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  static_assert(std::is_invocable_r_v<Message, Builder, NodeID>, "bad builder type");

  using PEGetter = decltype(pe_getter);
  static_assert(std::is_invocable_r_v<PEID, PEGetter, NodeID>, "bad pe getter type");

  PEID size, rank;
  std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

  // allocate message counters
  const PEID num_threads = omp_get_max_threads();
  std::vector<cache_aligned_vector<std::size_t>> num_messages(num_threads, cache_aligned_vector<std::size_t>(size));

  // count messages to each PE for each thread
  START_TIMER("Count messages", TIMER_FINE);
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
  STOP_TIMER(TIMER_FINE);

  // allocate send buffers
  START_TIMER("Allocation", TIMER_FINE);
  std::vector<Buffer> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) {
    send_buffers.emplace_back(num_messages.back()[pe]);
  }
  STOP_TIMER(TIMER_FINE);

  // fill buffers
  START_TIMER("Partition messages", TIMER_FINE);
#pragma omp parallel default(none) shared(pe_getter, send_buffers, size, from, to, builder, filter, graph, num_messages)
  {
    const PEID thread = omp_get_thread_num();
#pragma omp for
    for (NodeID u = from; u < to; ++u) {
      if (filter(u)) {
        const PEID pe = pe_getter(u);
        const auto slot = --num_messages[thread][pe];
        send_buffers[pe][slot] = builder(u);
      }
    }
  }
  STOP_TIMER(TIMER_FINE);

  sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator(),
                                   self);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_custom(const DistributedGraph &graph, const NodeID from, const NodeID to,
                                           auto &&filter, auto &&pe_getter, auto &&builder) {
  auto size = mpi::get_comm_size(graph.communicator());
  std::vector<Buffer> recv_buffers(size);
  sparse_alltoall_custom<Message, Buffer>(
      graph, from, to, std::forward<decltype(filter)>(filter), std::forward<decltype(pe_getter)>(pe_getter),
      std::forward<decltype(builder)>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); });
  return recv_buffers;
}
} // namespace dkaminpar::mpi::graph