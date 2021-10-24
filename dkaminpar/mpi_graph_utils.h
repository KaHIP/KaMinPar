/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/mpi_utils.h"
#include "kaminpar/datastructure/marker.h"

#include <tbb/concurrent_vector.h>
#include <type_traits>

#define SPARSE_ALLTOALL_NOFILTER [](NodeID) { return true; }

namespace dkaminpar::mpi::graph {
template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, SPARSE_ALLTOALL_NOFILTER,
                                                      std::forward<decltype(builder)>(builder),
                                                      std::forward<decltype(receiver)>(receiver));
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, auto &&filter, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, 0, graph.n(), std::forward<decltype(filter)>(filter),
                                                      std::forward<decltype(builder)>(builder),
                                                      std::forward<decltype(receiver)>(receiver));
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_ghost(const DistributedGraph &graph, const NodeID from, const NodeID to,
                                        auto &&filter, auto &&builder, auto &&receiver) {
  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  constexpr bool builder_invocable_with_pe = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID, PEID>;
  constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID>;
  static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, const Buffer<Message> &, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, const Buffer<Message> &>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // allocate send buffers
  std::vector<Buffer<Message>> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) { send_buffers.emplace_back(graph.edge_cut_to_pe(pe)); }

  // next free slot in send_buffers[]
  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);

  // fill send_buffers
  graph.pfor_nodes(from, to, [&](const NodeID u) {
    if (!filter(u)) { return; }

    for (const auto [e, v] : graph.neighbors(u)) {
      if (!graph.is_ghost_node(v)) { continue; }

      const PEID pe = graph.ghost_owner(v);
      const auto slot = next_message[pe]++;

      if constexpr (builder_invocable_with_pe) {
        send_buffers[pe][slot] = builder(u, e, v, pe);
      } else /* if (builder_invocable_without_pe) */ {
        send_buffers[pe][slot] = builder(u, e, v);
      }
    }
  });

  // resize filtered send buffer
  for (PEID pe = 0; pe < size; ++pe) { send_buffers[pe].resize(next_message[pe]); }

  sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator());
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
std::vector<Buffer<Message>> sparse_alltoall_interface_to_ghost_get(const DistributedGraph &graph, const NodeID from,
                                                                    const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer<Message>> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, from, to, std::forward<decltype(filter)>(filter),
                                                      std::forward<decltype(builder)>(builder),
                                                      [&](auto recv_buffer, const PEID pe) {
                                                        recv_buffers[pe] = std::move(recv_buffer);
                                                      });
  return recv_buffers;
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, SPARSE_ALLTOALL_NOFILTER,
                                                   std::forward<decltype(builder)>(builder),
                                                   std::forward<decltype(receiver)>(receiver));
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, auto &&filter, auto &&builder, auto &&receiver) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, 0, graph.n(), std::forward<decltype(filter)>(filter),
                                                   std::forward<decltype(builder)>(builder),
                                                   std::forward<decltype(receiver)>(receiver));
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_interface_to_pe(const DistributedGraph &graph, const NodeID from, const NodeID to, auto &&filter,
                                     auto &&builder, auto &&receiver) {
  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  constexpr bool builder_invocable_with_pe = std::is_invocable_r_v<Message, Builder, NodeID, PEID>;
  constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID>;
  static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

  PEID size, rank;
  std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

  // allocate send buffers
  std::vector<Buffer<Message>> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) { send_buffers.emplace_back(graph.comm_vol_to_pe(pe)); }

  // next free slot in send_buffers[]
  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);

  // fill send buffers
  graph.pfor_nodes_range(from, to, [&](const auto r) {
    shm::Marker<> created_message_for_pe(static_cast<std::size_t>(size));

    for (NodeID u = r.begin(); u < r.end(); ++u) {
      if (!filter(u)) { continue; }

      for (const NodeID v : graph.adjacent_nodes(u)) {
        if (graph.is_ghost_node(v)) {
          const PEID pe = graph.ghost_owner(v);
          ASSERT(static_cast<std::size_t>(pe) < send_buffers.size());

          if (!created_message_for_pe.get(pe)) {
            created_message_for_pe.set(pe);
            const auto slot = next_message[pe]++;
            ASSERT(static_cast<std::size_t>(slot) < send_buffers[pe].size());

            if constexpr (builder_invocable_with_pe) {
              send_buffers[pe][slot] = builder(u, pe);
            } else {
              send_buffers[pe][slot] = builder(u);
            }
          }
        }
      }
      created_message_for_pe.reset();
    }
  });

  // resize filtered send buffer
  for (PEID pe = 0; pe < size; ++pe) { send_buffers[pe].resize(next_message[pe]); }

  sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator());
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
std::vector<Buffer<Message>> sparse_alltoall_interface_to_pe_get(const DistributedGraph &graph, const NodeID from,
                                                                 const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer<Message>> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, from, to, std::forward<decltype(filter)>(filter),
                                                   std::forward<decltype(builder)>(builder),
                                                   [&](auto recv_buffer, const PEID pe) {
                                                     recv_buffers[pe] = std::move(recv_buffer);
                                                   });
  return recv_buffers;
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall_custom(const DistributedGraph &graph, const NodeID from, const NodeID to, auto &&filter,
                            auto &&builder, auto &&receiver) {
  using Filter = decltype(filter);
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");

  using Builder = decltype(builder);
  static_assert(std::is_invocable_r_v<std::pair<Message, PEID>, Builder, NodeID>, "bad builder type");

  PEID size, rank;
  std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

  // allocate send buffers
  std::vector<tbb::concurrent_vector<Message>> send_buffers(size);

  graph.pfor_nodes(from, to, [&](const NodeID u) {
    if (!filter(u)) { return; }
    const auto [message, pe] = builder(u);
    send_buffers[pe].push_back(std::move(message));
  });

  // TODO can we avoid copying here?
  std::vector<Buffer<Message>> real_send_buffers;
  for (PEID pe = 0; pe < size; ++pe) { real_send_buffers.emplace_back(send_buffers[pe].size()); }
  tbb::parallel_for(0, size, [&](const PEID pe) {
    std::copy(send_buffers[pe].begin(), send_buffers[pe].end(), real_send_buffers[pe].begin());
  });

  sparse_alltoall<Message, Buffer>(real_send_buffers, std::forward<decltype(receiver)>(receiver), graph.communicator());
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
std::vector<Buffer<Message>> sparse_alltoall_custom(const DistributedGraph &graph, const NodeID from, const NodeID to,
                                                    auto &&filter, auto &&builder) {
  auto size = mpi::get_comm_size(graph.communicator());
  std::vector<Buffer<Message>> recv_buffers(size);
  sparse_alltoall_custom<Message, Buffer>(graph, from, to, std::forward<decltype(filter)>(filter),
                                          std::forward<decltype(builder)>(builder),
                                          [&](auto recv_buffer, const PEID pe) {
                                            recv_buffers[pe] = std::move(recv_buffer);
                                          });
  return recv_buffers;
}
} // namespace dkaminpar::mpi::graph