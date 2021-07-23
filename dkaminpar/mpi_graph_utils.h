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

namespace dkaminpar::mpi {
template<typename Message>
using SendBuffer = std::vector<scalable_vector<Message>>;

template<typename Message, std::invocable<DNodeID, DEdgeID, DNodeID, PEID> Builder>
SendBuffer<Message> build_send_buffer_edge_cut(const DistributedGraph &graph, Builder &&builder) {
  // Allocate send buffers
  const PEID size = mpi::get_comm_size(graph.communicator());
  SendBuffer<Message> send_buffer(size);
  for (PEID pe = 0; pe < size; ++pe) { send_buffer.emplace_back(graph.edge_cut_to_pe(pe)); }

  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);

  // Create messages
  graph.pfor_nodes([&](const DNodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      if (graph.is_ghost_node(v)) {
        const PEID owner = graph.ghost_owner(v);
        ASSERT(owner < send_buffer.size());

        const std::size_t pos = next_message[owner]++;
        ASSERT(pos < send_buffer[owner].size());

        send_buffer[owner][pos] = builder(u, e, v, owner);
      }
    }
  });

  return send_buffer;
}

template<typename Message, std::invocable<DNodeID, PEID> Builder>
SendBuffer<Message> build_send_buffer_comm_vol(const DistributedGraph &graph, Builder &&builder) {
  // Allocate send buffers
  const PEID size = mpi::get_comm_size(graph.communicator());
  SendBuffer<Message> send_buffer(size);
  for (PEID pe = 0; pe < size; ++pe) { send_buffer.emplace_back(graph.comm_vol_to_pe(pe)); }

  std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(size);

  // Create messages
  graph.pfor_nodes_range([&](const auto r) {
    shm::Marker<> created_message_for_pe(static_cast<std::size_t>(size));

    for (DNodeID u = r.begin(); u < r.end(); ++u) {
      for (const DNodeID v : graph.adjacent_nodes(u)) {
        if (graph.is_ghost_node(v)) {
          const PEID owner = graph.ghost_owner(v);
          ASSERT(owner < send_buffer.size());

          if (!created_message_for_pe.get(owner)) {
            created_message_for_pe.set(owner);
            const std::size_t pos = next_message[owner]++;
            ASSERT(pos < send_buffer[owner].size());
            send_buffer[owner][pos] = builder(u, owner);
          }
        }
      }
      created_message_for_pe.reset();
    }
  });

  return send_buffer;
}
} // namespace dkaminpar::mpi