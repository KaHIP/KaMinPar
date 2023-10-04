/*******************************************************************************
 * @file:   distributed_graph_factories.h
 * @author: Daniel Seemaier
 * @date:   22.08.2022
 * @brief:  Factory functions for graph topologies that are commonly used in
 * unit tests.
 ******************************************************************************/
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include "tests/dist/distributed_graph_builder.h"

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::testing {
/*!
 * Creates a distributed path with `num_nodes_per_pe` nodes per PE.
 *
 * @param num_nodes_per_pe Number of nodes per PE.
 * @return Distributed graph with `num_nodes_per_pe` nodes per PE.
 */
inline DistributedGraph make_path(const NodeID num_nodes_per_pe) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  const NodeID n0 = num_nodes_per_pe * rank;

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(num_nodes_per_pe);

  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);
    if (n0 > 0 || u > 0) {
      builder.create_edge(1, n0 + u - 1);
    }
    if (rank + 1 < size || u + 1 < num_nodes_per_pe) {
      builder.create_edge(1, n0 + u + 1);
    }
  }

  return builder.finalize();
}

/*!
 * Creates a distributed circle with one node on each PE.
 *
 * @return Distributed graph with one node on each PE, nodes are connected in a
 * circle.
 */
inline DistributedGraph make_circle_graph() {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(1);

  const GlobalNodeID prev = static_cast<GlobalNodeID>(rank > 0 ? rank - 1 : size - 1);
  const GlobalNodeID next = static_cast<GlobalNodeID>((rank + 1) % size);

  builder.create_node(1);
  if (rank != static_cast<PEID>(prev)) {
    builder.create_edge(1, prev);
  }
  if (prev != next) {
    builder.create_edge(1, next);
  }

  return builder.finalize();
}

/*!
 * Creates a distributed graph with `num_nodes_per_pe` nodes per PE and zero
 * edges.
 *
 * @param num_nodes_per_pe Number of nodes on each PE.
 * @return Distributed graph with `num_nodes_per_pe` nodes per PE.
 */
inline DistributedGraph make_isolated_nodes_graph(const NodeID num_nodes_per_pe) {
  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(num_nodes_per_pe);
  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);
  }
  return builder.finalize();
}

/*!
 * Creates a distributed graph without any nodes.
 *
 * @return Distributed graph without any nodes.
 */
inline DistributedGraph make_empty_graph() {
  return make_isolated_nodes_graph(0);
}

/*!
 * Creates a distributed graph with `2 * num_edges_per_pe` nodes on each PE,
 * each pair connected by an edge.
 *
 * @param num_edges_per_pe Number of edges on each PE, with distinct endpoints.
 * @return Distributed graph with `2 * num_edges_per_pe` nodes and
 * `num_edges_per_pe` edges per PE.
 */
inline DistributedGraph make_isolated_edges_graph(const NodeID num_edges_per_pe) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const NodeID n0 = rank * num_edges_per_pe * 2;

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(2 * num_edges_per_pe);
  for (EdgeID e = 0; e < num_edges_per_pe; ++e) {
    builder.create_node(1);
    builder.create_edge(1, n0 + 2 * e + 1);
    builder.create_node(1);
    builder.create_edge(1, n0 + 2 * e);
  }
  return builder.finalize();
}

inline DistributedGraph make_local_complete_graph(const NodeID num_nodes_per_pe) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const GlobalNodeID n0 = rank * num_nodes_per_pe;

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(num_nodes_per_pe);
  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);
    for (NodeID v = 0; v < num_nodes_per_pe; ++v) {
      if (u != v) {
        builder.create_edge(1, n0 + v);
      }
    }
  }
  return builder.finalize();
}

inline DistributedGraph make_local_complete_bipartite_graph(const NodeID set_size_per_pe) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const GlobalNodeID n0 = rank * set_size_per_pe * 2;

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(2 * set_size_per_pe);
  for (NodeID u = 0; u < 2 * set_size_per_pe; ++u) {
    builder.create_node(1);

    for (NodeID v = 0; v < set_size_per_pe; ++v) {
      if (u < set_size_per_pe) {
        builder.create_edge(1, n0 + set_size_per_pe + v);
      } else {
        builder.create_edge(1, n0 + v);
      }
    }
  }
  return builder.finalize();
}

inline DistributedGraph make_global_complete_graph(const NodeID nodes_per_pe) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const GlobalNodeID n0 = rank * nodes_per_pe;

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(nodes_per_pe);
  for (NodeID u = 0; u < nodes_per_pe; ++u) {
    builder.create_node(1);

    for (GlobalNodeID v = 0; v < static_cast<GlobalNodeID>(size * nodes_per_pe); ++v) {
      if (n0 + u != v) {
        builder.create_edge(1, v);
      }
    }
  }
  return builder.finalize();
}

/*!
 * Creates a distributed graph with `num_nodes_per_pe` nodes on each PE.
 * The nodes on a single PE are connected to a clique.
 * Globally, nodes with the same local ID are connected to a circle.
 *
 * @param num_nodes_per_pe Number of nodes per PE.
 * @return Distributed graph with a clique on `num_nodes_per_pe` nodes on each
 * PE and `num_nodes_per_pe` global circles.
 */
inline DistributedGraph make_circle_clique_graph(const NodeID num_nodes_per_pe) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(num_nodes_per_pe);

  const GlobalNodeID my_n0 = rank * num_nodes_per_pe;
  const GlobalNodeID prev_n0 =
      (rank > 0 ? (rank - 1) * num_nodes_per_pe : (size - 1) * num_nodes_per_pe);
  const GlobalNodeID next_n0 = (rank + 1 < size ? (rank + 1) * num_nodes_per_pe : 0);

  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);

    // Clique
    for (NodeID v = 0; v < num_nodes_per_pe; ++v) {
      if (u == v) {
        continue;
      }
      builder.create_edge(1, my_n0 + v);
    }

    // Circle
    if (prev_n0 != my_n0) {
      builder.create_edge(1, prev_n0 + u);
    }
    if (next_n0 != prev_n0) {
      builder.create_edge(1, next_n0 + u);
    }
  }

  return builder.finalize();
}

/*!
 * Creates a distributed graph with `2 * num_nodes_per_pe` nodes on each PE,
 * that are connected to a node on the next / previous PE:
 *
 * O O-#-O O-#-O O
 * |   #######   |
 * +-------------+
 *
 * @param num_nodes_per_pe Number of nodes on each side of each PE.
 * @return Distributed graph as described above.
 */
inline DistributedGraph make_cut_edge_graph(const NodeID num_nodes_per_pe) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(2 * num_nodes_per_pe);

  const GlobalNodeID my_n0_to_prev = 2 * num_nodes_per_pe * rank;
  const GlobalNodeID my_n0_to_next = 2 * num_nodes_per_pe * rank + num_nodes_per_pe;

  // connect to prev PE
  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);

    GlobalNodeID neighbor;
    if (rank == 0) {
      neighbor = 2 * size * num_nodes_per_pe + my_n0_to_prev + u - num_nodes_per_pe;
    } else {
      neighbor = my_n0_to_prev + u - num_nodes_per_pe;
    }
    builder.create_edge(1, neighbor);
  }

  for (NodeID u = 0; u < num_nodes_per_pe; ++u) {
    builder.create_node(1);

    GlobalNodeID neighbor;
    if (rank + 1 == size) {
      neighbor = my_n0_to_next + u + num_nodes_per_pe - 2 * size * num_nodes_per_pe;
    } else {
      neighbor = my_n0_to_next + u + num_nodes_per_pe;
    }
    builder.create_edge(1, neighbor);
  }

  return builder.finalize();
}
} // namespace kaminpar::dist::testing
