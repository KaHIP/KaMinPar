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

#include <concepts>

namespace dkaminpar::mpi::graph {
namespace internal {
template<typename Message, template<typename> typename Buffer, std::invocable<PEID, const Buffer<Message> &> Receiver>
void perform_sparse_alltoall(const std::vector<Buffer<Message>> &send_buffers, Receiver &&recv_lambda, const PEID size,
                             const PEID rank, const int tag, MPI_Comm comm) {
  std::vector<MPI_Request> requests;
  requests.reserve(size);

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      requests.emplace_back();
      mpi::isend(send_buffers[pe], pe, tag, requests.back(), comm);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      const auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, tag, MPI_STATUS_IGNORE, comm);
      recv_lambda(pe, recv_buffer);
    }
  }

  mpi::waitall(requests);
}
} // namespace internal

template<typename Message, template<typename> typename Buffer = scalable_vector,
         std::invocable<NodeID, EdgeID, NodeID, PEID> Builder, std::invocable<PEID, const Buffer<Message> &> Receiver>
void sparse_alltoall_ghost_edge(const DistributedGraph &graph, Builder &&builder_lambda, Receiver &&recv_lambda,
                                const int tag = 0) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // allocate send buffers
  std::vector<Buffer<Message>> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) { send_buffers.emplace_back(graph.edge_cut_to_pe(pe)); }

  // create messages for send buffer
  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);
  graph.pfor_nodes([&](const NodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      if (graph.is_ghost_node(v)) {
        const PEID pe = graph.ghost_owner(v);
        send_buffers[pe][next_message[pe]++] = builder_lambda(u, e, v, pe);
      }
    }
  });

  internal::perform_sparse_alltoall<Message, Buffer, Receiver>(send_buffers, std::forward<Receiver>(recv_lambda), size,
                                                               rank, tag, graph.communicator());
}

template<typename Message, template<typename> typename Buffer, std::invocable<NodeID> Filter,
         std::invocable<NodeID, PEID> Builder, std::invocable<PEID, const Buffer<Message> &> Receiver>
void sparse_alltoall_interface_node_range_filtered(const DistributedGraph &graph, const NodeID from, const NodeID to,
                                          Filter &&filter_lambda, Builder &&builder_lambda, Receiver &&recv_lambda,
                                          const int tag = 0) {
  PEID size, rank;
  std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

  // allocate send buffers
  std::vector<Buffer<Message>> send_buffers;
  for (PEID pe = 0; pe < size; ++pe) { send_buffers.emplace_back(graph.comm_vol_to_pe(pe)); }

  // Create messages
  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);
  graph.pfor_nodes_range(from, to, [&](const auto r) {
    shm::Marker<> created_message_for_pe(static_cast<std::size_t>(size));

    for (NodeID u = r.begin(); u < r.end(); ++u) {
      if (!filter_lambda(u)) { continue; }

      for (const NodeID v : graph.adjacent_nodes(u)) {
        if (graph.is_ghost_node(v)) {
          const PEID pe = graph.ghost_owner(v);
          ASSERT(static_cast<std::size_t>(pe) < send_buffers.size());

          if (!created_message_for_pe.get(pe)) {
            created_message_for_pe.set(pe);
            const auto pos = next_message[pe]++;
            ASSERT(static_cast<std::size_t>(pos) < send_buffers[pe].size())
                << V(pe) << V(pos) << V(send_buffers[pe].size());
            send_buffers[pe][pos] = builder_lambda(u, pe);
          }
        }
      }
      created_message_for_pe.reset();
    }
  });

  internal::perform_sparse_alltoall<Message, Buffer, Receiver>(send_buffers, std::forward<Receiver>(recv_lambda), size,
                                                               rank, tag, graph.communicator());
}

template<typename Message, template<typename> typename Buffer, std::invocable<NodeID> Filter,
         std::invocable<NodeID, PEID> Builder, std::invocable<PEID, const Buffer<Message> &> Receiver>
void sparse_alltoall_interface_node_filtered(const DistributedGraph &graph, Filter &&filter_lambda, Builder &&builder_lambda,
                                    Receiver &&recv_lambda, const int tag = 0) {
  sparse_alltoall_interface_node_range_filtered<Message, Buffer>(graph, 0, graph.n(), std::forward<Filter>(filter_lambda),
                                       std::forward<Builder>(builder_lambda), std::forward<Receiver>(recv_lambda), tag);
}

// overload without filter lambda
template<typename Message, template<typename> typename Buffer, std::invocable<NodeID, PEID> Builder,
         std::invocable<PEID, const Buffer<Message> &> Receiver>
void sparse_alltoall_interface_node(const DistributedGraph &graph, Builder &&builder_lambda, Receiver &&recv_lambda,
                                    const int tag = 0) {
  sparse_alltoall_interface_node_filtered<Message, Buffer>(
      graph, [](const NodeID) { return true; }, std::forward<Builder>(builder_lambda),
      std::forward<Receiver>(recv_lambda), tag);
}
} // namespace dkaminpar::mpi::graph