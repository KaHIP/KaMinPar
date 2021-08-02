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
#include "dkaminpar/distributed_io.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/utility/distributed_math.h"
#include "kaminpar/io.h"
#include "mpi_utils.h"

#include <io.h>

namespace dkaminpar::io {
SET_DEBUG(false);

namespace metis {
namespace shm = kaminpar::io::metis;

DistributedGraph read_node_balanced(const std::string &filename) {
  const auto comm_info = mpi::get_comm_info();
  const PEID size = comm_info.first;
  const PEID rank = comm_info.second;

  graph::Builder builder{};

  GlobalNodeID current = 0;
  GlobalNodeID from = 0;
  GlobalNodeID to = 0;

  shm::read_observable(
      filename,
      [&](const auto &format) {
        const auto global_n = static_cast<GlobalNodeID>(format.number_of_nodes);
        const auto global_m = static_cast<GlobalNodeID>(format.number_of_edges) * 2;
        DBG << "Loading graph with global_n=" << global_n << " and global_m=" << global_m;

        scalable_vector<GlobalNodeID> node_distribution(size + 1);
        for (PEID p = 0; p < size; ++p) {
          const auto [from, to] = math::compute_local_range<GlobalNodeID>(global_n, size, p);
          node_distribution[p + 1] = to;
        }
        ASSERT(node_distribution.front() == 0);
        ASSERT(node_distribution.back() == global_n);

        from = node_distribution[rank];
        to = node_distribution[rank + 1];
        DBG << "PE " << rank << ": from=" << from << " to=" << to << " n=" << format.number_of_nodes;

        builder.initialize(global_n, global_m, rank, std::move(node_distribution));
      },
      [&](const std::uint64_t &u_weight) {
        ++current;
        if (current > to) { return false; }
        if (current > from) { builder.create_node(static_cast<NodeWeight>(u_weight)); }
        return true;
      },
      [&](const std::uint64_t &e_weight, const std::uint64_t &v) {
        if (current > from) { builder.create_edge(static_cast<EdgeWeight>(e_weight), static_cast<GlobalNodeID>(v)); }
      });

  return builder.finalize();
}

DistributedGraph read_edge_balanced(const std::string &filename) {
  UNUSED(filename);
  return {};
}

void write(const std::string &filename, const DistributedGraph &graph, const bool write_node_weights,
           const bool write_edge_weights) {
  { std::ofstream tmp(filename); } // clear file

  mpi::sequentially([&](const PEID pe) {
    DLOG << "start " << pe;
    std::ofstream out(filename, std::ios_base::out | std::ios_base::app);
    if (pe == 0) {
      out << graph.global_n() << " " << graph.global_m() / 2;
      if (write_node_weights || write_edge_weights) {
        out << " ";
        out << static_cast<int>(write_node_weights);
        out << static_cast<int>(write_edge_weights);
      }
      out << "\n";
    }

    for (const NodeID u : graph.nodes()) {
      if (write_node_weights) { out << graph.node_weight(u) << " "; }
      for (const auto [e, v] : graph.neighbors(u)) {
        out << graph.local_to_global_node(v) + 1 << " ";
        if (write_edge_weights) { out << graph.edge_weight(e) << " "; }
      }
      out << "\n";
    }
    DLOG << "end " << pe;
  });
}
} // namespace metis
} // namespace dkaminpar::io