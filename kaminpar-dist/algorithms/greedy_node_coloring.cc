/*******************************************************************************
 * Basic implementation of a distributed vertex coloring algorithm.
 *
 * @file:   greedy_node_coloring.cc
 * @author: Daniel Seemaier
 * @date:   11.11.2022
 ******************************************************************************/
#include "kaminpar-dist/algorithms/greedy_node_coloring.h"

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(false);
}

NoinitVector<ColorID> compute_node_coloring_sequentially(
    const DistributedGraph &graph, const NodeID number_of_supersteps
) {
  KASSERT(number_of_supersteps > 0u, "bad parameter", assert::light);
  SCOPED_TIMER("Compute greedy node coloring");

  // Initialize coloring to 0 == no color picked yet
  NoinitVector<ColorID> coloring(graph.total_n());
  graph.pfor_all_nodes([&](const NodeID u) { coloring[u] = 0; });

  // Use max degree in the graph as an upper bound on the number of colors
  // required
  TransformedIotaRange degrees(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    return graph.degree(u);
  });
  const EdgeID max_degree = parallel::max_element(degrees.begin(), degrees.end());
  const ColorID max_colors = mpi::allreduce(max_degree, MPI_MAX, graph.communicator()) + 1;

  // Marker to keep track of the colors already incident to the current node
  Marker<> incident_colors(max_colors);

  // Keep track of nodes that still need a color
  NoinitVector<std::uint8_t> active(graph.n());
  graph.pfor_nodes([&](const NodeID u) { active[u] = 1; });

  bool converged;
  do {
    converged = true;

    for (NodeID superstep = 0; superstep < number_of_supersteps; ++superstep) {
      const auto [from, to] = math::compute_local_range(graph.n(), number_of_supersteps, superstep);

      // Color all nodes in [from, to)
      for (const NodeID u : graph.nodes(from, to)) {
        if (!active[u]) {
          continue;
        }

        bool is_interface_node = false;
        for (const auto [e, v] : graph.neighbors(u)) {
          is_interface_node = is_interface_node || graph.is_ghost_node(v);

          // @todo replace v < u with random numbers r(v) < r(u)
          if (coloring[v] != 0 && (coloring[u] == 0 || !(coloring[v] == coloring[u] &&
                                                         graph.local_to_global_node(u) <
                                                             graph.local_to_global_node(v)))) {
            incident_colors.set<true>(coloring[v] - 1);
          }
        }

        if (coloring[u] == 0) {
          coloring[u] = incident_colors.first_unmarked_element() + 1;
          DBGC(u == 156543 || u == 262712) << "setting " << u << " to " << coloring[u] << " A";
          if (!is_interface_node) {
            active[u] = 0;
          }
        } else if (incident_colors.get(coloring[u] - 1)) {
          coloring[u] = incident_colors.first_unmarked_element() + 1;
          DBGC(u == 156543 || u == 262712 || graph.local_to_global_node(u) == 681015)
              << "setting " << u << " to " << coloring[u] << " B, global "
              << graph.local_to_global_node(u);
        } else {
          active[u] = 0;
        }

        incident_colors.reset();
      }

      // Synchronize coloring of interface <-> ghost nodes
      struct Message {
        NodeID node;
        ColorID color;
      };

      mpi::graph::sparse_alltoall_interface_to_pe<Message>(
          graph,
          from,
          to,
          [&](const NodeID u) { return active[u]; },
          [&](const NodeID u) -> Message {
            DBGC(u == 156543) << "Sending " << u << " --> " << coloring[u];
            return {.node = u, .color = coloring[u]};
          },
          [&](const auto &recv_buffer, const PEID pe) {
            converged &= recv_buffer.empty();
            tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
              const auto [local_node_on_pe, color] = recv_buffer[i];
              const GlobalNodeID global_node =
                  static_cast<GlobalNodeID>(graph.offset_n(pe) + local_node_on_pe);
              const NodeID local_node = graph.global_to_local_node(global_node);
              coloring[local_node] = color;
              DBGC(local_node == 156543 || local_node == 262712)
                  << "setting " << local_node << " to " << coloring[local_node] << " C, global "
                  << graph.local_to_global_node(local_node);
            });
          }
      );
    }
  } while (!mpi::allreduce(converged, MPI_LAND, graph.communicator()));

  // Check that all nodes have a color assigned (i.e., coloring[u] >= 1)
  KASSERT(
      [&] {
        for (const NodeID u : graph.all_nodes()) {
          if (coloring[u] == 0) {
            return false;
          }
        }
        return true;
      }(),
      "node coloring is incomplete",
      assert::heavy
  );

  // Check that adjacent nodes have different colores
  KASSERT(
      [&] {
        for (const NodeID u : graph.nodes()) {
          for (const auto v : graph.adjacent_nodes(u)) {
            if (coloring[u] == coloring[v]) {
              LOG_WARNING << "bad color for node " << u << " with neighbor " << v << ": "
                          << coloring[u];
              return false;
            }
          }
        }
        return true;
      }(),
      "local node coloring is invalid",
      assert::heavy
  );

  // Check that interface and ghost nodes have the same colors
  KASSERT(
      [&] {
        struct Message {
          GlobalNodeID node;
          ColorID color;
        };
        bool inconsistent = false;
        mpi::graph::sparse_alltoall_interface_to_pe<Message>(
            graph,
            [&](const NodeID u) -> Message {
              return {.node = graph.local_to_global_node(u), .color = coloring[u]};
            },
            [&](const auto &recv_buffer) {
              tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
                const auto [node, color] = recv_buffer[i];
                const NodeID local_node = graph.global_to_local_node(node);
                if (coloring[local_node] != color) {
                  inconsistent = true;
                }
              });
            }
        );
        return !inconsistent;
      }(),
      "global node coloring inconsistent",
      assert::heavy
  );

  // Make colors start at 0
  tbb::parallel_for<NodeID>(0, graph.total_n(), [&](const NodeID u) { coloring[u] -= 1; });

  return coloring;
}
} // namespace kaminpar::dist
