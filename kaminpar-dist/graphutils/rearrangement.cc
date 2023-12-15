/*******************************************************************************
 * Sort and rearrange a graph by degree buckets.
 *
 * @file:   graph_rearrangement.cc
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 ******************************************************************************/
#include "kaminpar-dist/graphutils/rearrangement.h"

#include "communication.h"

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist::graph {
DistributedGraph rearrange(DistributedGraph graph, const Context &ctx) {
  if (ctx.rearrange_by == GraphOrdering::NATURAL) {
    // nothing to do
  } else if (ctx.rearrange_by == GraphOrdering::DEGREE_BUCKETS) {
    graph = graph::rearrange_by_degree_buckets(std::move(graph));
  } else if (ctx.rearrange_by == GraphOrdering::COLORING) {
    graph = graph::rearrange_by_coloring(std::move(graph), ctx);
  }

  KASSERT(
      debug::validate_graph(graph),
      "input graph verification failed after rearranging graph",
      assert::heavy
  );
  return graph;
}

DistributedGraph rearrange_by_degree_buckets(DistributedGraph graph) {
  SCOPED_TIMER("Rearrange graph", "By degree buckets");
  auto permutations = shm::graph::sort_by_degree_buckets<false>(graph.raw_nodes());
  return rearrange_by_permutation(
      std::move(graph),
      std::move(permutations.old_to_new),
      std::move(permutations.new_to_old),
      false
  );
}

DistributedGraph rearrange_by_coloring(DistributedGraph graph, const Context &ctx) {
  SCOPED_TIMER("Rearrange graph", "By coloring");

  auto coloring = compute_node_coloring_sequentially(
      graph, ctx.refinement.colored_lp.coloring_chunks.compute(ctx.parallel)
  );
  const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
  const ColorID num_colors = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());

  START_TIMER("Allocation");
  StaticArray<NodeID> old_to_new(graph.n());
  StaticArray<NodeID> new_to_old(graph.n());
  StaticArray<NodeID> color_sizes(num_colors + 1);
  STOP_TIMER();

  TIMED_SCOPE("Count color sizes") {
    graph.pfor_nodes([&](const NodeID u) {
      const ColorID c = coloring[u];
      KASSERT(c < num_colors);
      __atomic_fetch_add(&color_sizes[c], 1, __ATOMIC_RELAXED);
    });
    parallel::prefix_sum(color_sizes.begin(), color_sizes.end(), color_sizes.begin());
  };

  TIMED_SCOPE("Sort nodes") {
    graph.pfor_nodes([&](const NodeID u) {
      const ColorID c = coloring[u];
      const std::size_t i = __atomic_sub_fetch(&color_sizes[c], 1, __ATOMIC_SEQ_CST);
      old_to_new[u] = i;
      new_to_old[i] = u;
    });
  };

  graph = rearrange_by_permutation(
      std::move(graph), std::move(old_to_new), std::move(new_to_old), false
  );
  graph.set_color_sorted(std::move(color_sizes));
  return graph;
}

DistributedGraph rearrange_by_permutation(
    DistributedGraph graph,
    StaticArray<NodeID> old_to_new,
    StaticArray<NodeID> new_to_old,
    const bool degree_sorted
) {
  shm::graph::NodePermutations<StaticArray> permutations{
      std::move(old_to_new), std::move(new_to_old)};

  const auto &old_nodes = graph.raw_nodes();
  const auto &old_edges = graph.raw_edges();
  const auto &old_node_weights = graph.raw_node_weights();
  const auto &old_edge_weights = graph.raw_edge_weights();

  // rearrange nodes, edges, node weights and edge weights
  // ghost nodes are copied without remapping them to new IDs
  START_TIMER("Allocation");
  StaticArray<EdgeID> new_nodes(old_nodes.size());
  StaticArray<NodeID> new_edges(old_edges.size());
  StaticArray<NodeWeight> new_node_weights(old_node_weights.size());
  StaticArray<EdgeWeight> new_edge_weights(old_edge_weights.size());
  STOP_TIMER();

  shm::graph::build_permuted_graph<StaticArray, true, NodeID, EdgeID, NodeWeight, EdgeWeight>(
      old_nodes,
      old_edges,
      old_node_weights,
      old_edge_weights,
      permutations,
      new_nodes,
      new_edges,
      new_node_weights,
      new_edge_weights
  );

  // copy weight of ghost nodes
  if (!new_node_weights.empty()) {
    tbb::parallel_for<NodeID>(graph.n(), graph.total_n(), [&](const NodeID u) {
      new_node_weights[u] = old_node_weights[u];
    });
  }

  // communicate new global IDs of ghost nodes
  struct ChangedNodeLabel {
    NodeID old_node_local;
    NodeID new_node_local;
  };

  auto received = mpi::graph::sparse_alltoall_interface_to_pe_get<ChangedNodeLabel>(
      graph,
      [&](const NodeID u) -> ChangedNodeLabel {
        return {.old_node_local = u, .new_node_local = permutations.old_to_new[u]};
      }
  );

  const NodeID n = graph.n();
  auto old_global_to_ghost = graph.take_global_to_ghost(); // TODO cannot be cleared?
  growt::StaticGhostNodeMapping new_global_to_ghost(old_global_to_ghost.capacity());
  auto new_ghost_to_global = graph.take_ghost_to_global(); // can be reused

  parallel::chunked_for(received, [&](const ChangedNodeLabel &message, const PEID pe) {
    const auto &[old_node_local, new_node_local] = message;
    const GlobalNodeID old_node_global = graph.offset_n(pe) + old_node_local;
    const GlobalNodeID new_node_global = graph.offset_n(pe) + new_node_local;

    KASSERT(old_global_to_ghost.find(old_node_global + 1) != old_global_to_ghost.end());
    const NodeID ghost_node = (*old_global_to_ghost.find(old_node_global + 1)).second;
    new_global_to_ghost.insert(new_node_global + 1, ghost_node);
    new_ghost_to_global[ghost_node - n] = new_node_global;
  });

  DistributedGraph new_graph(
      graph.take_node_distribution(),
      graph.take_edge_distribution(),
      std::move(new_nodes),
      std::move(new_edges),
      std::move(new_node_weights),
      std::move(new_edge_weights),
      graph.take_ghost_owner(),
      std::move(new_ghost_to_global),
      std::move(new_global_to_ghost),
      degree_sorted,
      graph.communicator()
  );
  new_graph.set_permutation(std::move(permutations.old_to_new));
  return new_graph;
}
} // namespace kaminpar::dist::graph
