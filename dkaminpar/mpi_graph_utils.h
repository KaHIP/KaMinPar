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
#include "kaminpar/datastructure/marker.h"

#include <type_traits>

namespace dkaminpar::mpi::graph {
namespace internal {
template<typename Message, template<typename> typename Buffer>
void perform_sparse_alltoall(const std::vector<Buffer<Message>> &send_buffers, auto &&receiver, const PEID size,
                             const PEID rank, MPI_Comm comm) {
  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, Buffer<Message>, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer<Message>>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  std::vector<MPI_Request> requests;
  requests.reserve(size);

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      requests.emplace_back();
      mpi::isend(send_buffers[pe], pe, 0, requests.back(), comm);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      const auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, MPI_STATUS_IGNORE, comm);
      if constexpr (receiver_invocable_with_pe) {
        receiver(std::move(recv_buffer), pe);
      } else /* if (receiver_invocable_without_pe) */ {
        receiver(std::move(recv_buffer));
      }
    }
  }

  mpi::waitall(requests);
}
} // namespace internal

#define SPARSE_ALLTOALL_NOFILTER [](NodeID) { return true; }

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

  internal::perform_sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), size,
                                                     rank, graph.communicator());
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
std::vector<Buffer<Message>> sparse_alltoall_interface_to_ghost_get(const DistributedGraph &graph, const NodeID from,
                                                                    const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer<Message>> recv_buffers;
  sparse_alltoall_interface_to_ghost<Message, Buffer>(graph, from, to, std::forward<decltype(filter)>(filter),
                                                      std::forward<decltype(builder)>(builder), [&](auto recv_buffer) {
                                                        recv_buffers.push_back(std::move(recv_buffer));
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

  internal::perform_sparse_alltoall<Message, Buffer>(send_buffers, std::forward<decltype(receiver)>(receiver), size,
                                                     rank, graph.communicator());
}

template<typename Message, template<typename> typename Buffer = scalable_vector>
std::vector<Buffer<Message>> sparse_alltoall_interface_to_pe_get(const DistributedGraph &graph, const NodeID from,
                                                                 const NodeID to, auto &&filter, auto &&builder) {
  std::vector<Buffer<Message>> recv_buffers;
  sparse_alltoall_interface_to_pe<Message, Buffer>(graph, from, to, std::forward<decltype(filter)>(filter),
                                                   std::forward<decltype(builder)>(builder), [&](auto recv_buffer) {
                                                     recv_buffers.push_back(std::move(recv_buffer));
                                                   });
  return recv_buffers;
}
} // namespace dkaminpar::mpi::graph