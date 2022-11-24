/*******************************************************************************
 * @file:   graph_rearrangement.cc
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 * @brief:  Sort and rearrange a graph by degree buckets.
 ******************************************************************************/
#include "dkaminpar/graphutils/graph_rearrangement.h"

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/mpi/graph_communication.h"

#include "kaminpar/graphutils/graph_permutation.h"
#include "kaminpar/graphutils/graph_rearrangement.h"

#include "common/datastructures/marker.h"
#include "common/parallel/atomic.h"
#include "common/parallel/loops.h"
#include "common/timer.h"

namespace kaminpar::dist::graph {
DistributedGraph rearrange_by_degree_buckets(DistributedGraph graph) {
    SCOPED_TIMER("Rearrange graph", "By degree buckets");
    auto permutations = shm::graph::sort_by_degree_buckets<scalable_vector, false>(graph.raw_nodes());
    return rearrange_by_permutation(
        std::move(graph), std::move(permutations.old_to_new), std::move(permutations.new_to_old)
    );
}

DistributedGraph rearrange_by_coloring(DistributedGraph graph, const Context& ctx) {
    SCOPED_TIMER("Rearrange graph", "By coloring");

    auto          coloring = compute_node_coloring_sequentially(graph, ctx.refinement.colored_lp.num_coloring_chunks);
    const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    const ColorID num_colors       = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());

    START_TIMER("Allocation");
    scalable_vector<NodeID> old_to_new(graph.n());
    scalable_vector<NodeID> new_to_old(graph.n());
    scalable_vector<NodeID> color_sizes(num_colors + 1);
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
            const ColorID     c = coloring[u];
            const std::size_t i = __atomic_sub_fetch(&color_sizes[c], 1, __ATOMIC_SEQ_CST);
            old_to_new[u]       = i;
            new_to_old[i]       = u;
        });
    };

    graph = rearrange_by_permutation(std::move(graph), std::move(old_to_new), std::move(new_to_old));
    graph.set_color_sorted(std::move(color_sizes));
    return graph;
}

DistributedGraph rearrange_by_permutation(
    DistributedGraph graph, scalable_vector<NodeID> old_to_new, scalable_vector<NodeID> new_to_old
) {
    shm::graph::NodePermutations<scalable_vector> permutations{std::move(old_to_new), std::move(new_to_old)};

    const auto& old_nodes        = graph.raw_nodes();
    const auto& old_edges        = graph.raw_edges();
    const auto& old_node_weights = graph.raw_node_weights();
    const auto& old_edge_weights = graph.raw_edge_weights();

    // rearrange nodes, edges, node weights and edge weights
    // ghost nodes are copied without remapping them to new IDs
    START_TIMER("Allocation");
    scalable_vector<EdgeID>     new_nodes(old_nodes.size());
    scalable_vector<NodeID>     new_edges(old_edges.size());
    scalable_vector<NodeWeight> new_node_weights(old_node_weights.size());
    scalable_vector<EdgeWeight> new_edge_weights(old_edge_weights.size());
    STOP_TIMER();

    shm::graph::build_permuted_graph<scalable_vector, true, NodeID, EdgeID, NodeWeight, EdgeWeight>(
        old_nodes, old_edges, old_node_weights, old_edge_weights, permutations, new_nodes, new_edges, new_node_weights,
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

    const NodeID                  n                   = graph.n();
    auto                          old_global_to_ghost = graph.take_global_to_ghost(); // TODO cannot be cleared?
    growt::StaticGhostNodeMapping new_global_to_ghost(old_global_to_ghost.capacity());
    auto                          new_ghost_to_global = graph.take_ghost_to_global(); // can be reused

    parallel::chunked_for(received, [&](const ChangedNodeLabel& message, const PEID pe) {
        const auto& [old_node_local, new_node_local] = message;
        const GlobalNodeID old_node_global           = graph.offset_n(pe) + old_node_local;
        const GlobalNodeID new_node_global           = graph.offset_n(pe) + new_node_local;

        KASSERT(old_global_to_ghost.find(old_node_global + 1) != old_global_to_ghost.end());
        const NodeID ghost_node = (*old_global_to_ghost.find(old_node_global + 1)).second;
        new_global_to_ghost.insert(new_node_global + 1, ghost_node);
        new_ghost_to_global[ghost_node - n] = new_node_global;
    });

    return {
        graph.take_node_distribution(),
        graph.take_edge_distribution(),
        std::move(new_nodes),
        std::move(new_edges),
        std::move(new_node_weights),
        std::move(new_edge_weights),
        graph.take_ghost_owner(),
        std::move(new_ghost_to_global),
        std::move(new_global_to_ghost),
        true,
        graph.communicator()};
}
} // namespace kaminpar::dist::graph
