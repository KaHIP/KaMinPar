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
#include "dkaminpar/mpi_wrapper.h"
#include "dkaminpar/utility/math.h"
#include "kaminpar/io.h"

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
          const auto [p_from, p_to] = math::compute_local_range<GlobalNodeID>(global_n, size, p);
          node_distribution[p + 1] = p_to;
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
  const auto comm_info = mpi::get_comm_info();
  const PEID size = comm_info.first;
  const PEID rank = comm_info.second;

  PEID current_pe = 0;
  GlobalNodeID current_node = 0;
  GlobalEdgeID current_edge = 0;
  GlobalEdgeID to = 0;

  scalable_vector<EdgeID> nodes;
  scalable_vector<GlobalNodeID> global_edges;
  scalable_vector<NodeWeight> node_weights;
  scalable_vector<EdgeWeight> edge_weights;
  scalable_vector<PEID> ghost_owner;
  scalable_vector<GlobalNodeID> ghost_to_global;
  std::unordered_map<GlobalNodeID, NodeID> global_to_ghost;

  scalable_vector<GlobalNodeID> node_distribution(size + 1);
  scalable_vector<GlobalEdgeID> edge_distribution(size + 1);

  // read graph file
  shm::read_observable(
      filename,
      [&](const auto &format) {
        const auto global_n = static_cast<GlobalNodeID>(format.number_of_nodes);
        const auto global_m = static_cast<GlobalEdgeID>(format.number_of_edges) * 2;
        node_distribution.back() = global_n;
        edge_distribution.back() = global_m;
        const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
        to = pe_to;
      },
      [&](const std::uint64_t &u_weight) {
        if (current_edge >= to) {
          node_distribution[current_pe] = current_node;
          edge_distribution[current_pe] = current_edge;
          ++current_pe;

          const GlobalEdgeID global_m = edge_distribution.back();
          const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
          to = pe_to;
        }

        if (current_pe == rank) {
          nodes.push_back(global_edges.size());
          node_weights.push_back(static_cast<NodeWeight>(u_weight));
        }

        ++current_node;
        return true;
      },
      [&](const std::uint64_t &e_weight, const std::uint64_t &v) {
        if (current_pe == rank) {
          global_edges.push_back(static_cast<GlobalNodeID>(v));
          edge_weights.push_back(static_cast<EdgeWeight>(e_weight));
        }
        ++current_edge;
      });

  // at this point we should have a valid node and edge distribution
  const GlobalNodeID offset_n = node_distribution[rank];
  const auto local_n = static_cast<NodeID>(node_distribution[rank + 1] - node_distribution[rank]);

  // remap global edges to local edges and create ghost PEs
  scalable_vector<NodeID> edges(global_edges.size());
  for (std::size_t i = 0; i < global_edges.size(); ++i) {
    const GlobalNodeID global_v = global_edges[i];
    if (offset_n <= global_v && global_v < offset_n + local_n) { // owned node
      edges[i] = static_cast<NodeID>(global_v - offset_n);
    } else { // ghost node
      if (!global_to_ghost.contains(global_v)) {
        const NodeID local_id = local_n + ghost_to_global.size();
        ghost_to_global.push_back(global_v);
        global_to_ghost[global_v] = local_id;

        auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), global_v);
        const auto owner = static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
        ghost_owner.push_back(owner);
      }

      edges[i] = global_to_ghost[global_v];
    }
  }

  // init graph
  return {std::move(node_distribution),
          std::move(edge_distribution),
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          std::move(ghost_owner),
          std::move(ghost_to_global),
          std::move(global_to_ghost),
          MPI_COMM_WORLD};
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