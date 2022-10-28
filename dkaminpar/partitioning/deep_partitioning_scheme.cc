/*******************************************************************************
 * @file:   deep_partitioning.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Deep multilevel graph partitioning scheme.
 ******************************************************************************/
#include "dkaminpar/partitioning/deep_partitioning_scheme.h"

#include <cmath>
#include <iomanip>

#include <mpi.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/debug.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/graphutils/graph_extraction.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/partitioning/kway_partitioning_scheme.h"
#include "dkaminpar/refinement/balancer.h"

#include "kaminpar/datastructures/graph.h"

#include "common/assertion_levels.h"
#include "common/math.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_DEBUG(false);

DeepPartitioningScheme::DeepPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {
    _coarseners.emplace(_input_graph, _input_ctx);
}

void DeepPartitioningScheme::print_coarsening_level(const GlobalNodeWeight max_cluster_weight) const {
    const auto* coarsener = get_current_coarsener();
    const auto* graph     = coarsener->coarsest();
    mpi::barrier(graph->communicator());

    SCOPED_TIMER("Print coarse graph statistics");

    const auto [n_min, n_avg, n_max, n_sum] = mpi::gather_statistics(graph->n(), graph->communicator());
    const double n_imbalance                = 1.0 * n_max / n_avg;
    const auto [ghost_n_min, ghost_n_avg, ghost_n_max, ghost_n_sum] =
        mpi::gather_statistics(graph->ghost_n(), graph->communicator());
    const double ghost_n_imbalance          = 1.0 * ghost_n_max / ghost_n_avg;
    const auto [m_min, m_avg, m_max, m_sum] = mpi::gather_statistics(graph->m(), graph->communicator());
    const double m_imbalance                = 1.0 * m_max / m_avg;
    const auto [max_node_weight_min, max_node_weight_avg, max_node_weight_max, max_node_weight_sum] =
        mpi::gather_statistics(graph->max_node_weight(), graph->communicator());
    const auto max_value = std::max(
        {static_cast<GlobalNodeID>(n_max), static_cast<GlobalNodeID>(ghost_n_max), static_cast<GlobalNodeID>(m_max),
         static_cast<GlobalNodeID>(max_node_weight_max)}
    );
    const int width = std::log10(max_value) + 1;

    LOG << "Coarsening -> Level " << coarsener->level() << ":";
    LOG << "  Number of local nodes: [Min=" << std::setw(width) << n_min << " | Mean=" << std::setw(width)
        << static_cast<NodeID>(n_avg) << " | Max=" << std::setw(width) << n_max
        << " | Imbalance=" << std::setprecision(2) << std::setw(width) << n_imbalance << "]";
    LOG << "  Number of ghost nodes: [Min=" << std::setw(width) << ghost_n_min << " | Mean=" << std::setw(width)
        << static_cast<NodeID>(ghost_n_avg) << " | Max=" << std::setw(width) << ghost_n_max
        << " | Imbalance=" << std::setprecision(2) << std::setw(width) << ghost_n_imbalance << "]";
    LOG << "  Number of edges:       [Min=" << std::setw(width) << m_min << " | Mean=" << std::setw(width)
        << static_cast<EdgeID>(m_avg) << " | Max=" << std::setw(width) << m_max
        << " | Imbalance=" << std::setprecision(2) << std::setw(width) << m_imbalance << "]";
    LOG << "  Maximum node weight:   [Min=" << std::setw(width) << max_node_weight_min << " | Mean=" << std::setw(width)
        << static_cast<NodeWeight>(max_node_weight_avg) << " | Max=" << std::setw(width) << max_node_weight_max << "]";
    LOG << "    Expected: " << max_cluster_weight;
    LOG;

    mpi::barrier(graph->communicator());
}

void DeepPartitioningScheme::print_coarsening_converged() const {
    LOG << "==> Coarsening converged.";
    LOG;
}

void DeepPartitioningScheme::print_coarsening_terminated(const GlobalNodeID desired_num_nodes) const {
    LOG << "==> Coarsening terminated with less than " << desired_num_nodes << " nodes.";
    LOG;
}

void DeepPartitioningScheme::print_initial_partitioning_result(
    const DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx
) const {
    mpi::barrier(p_graph.communicator());

    SCOPED_TIMER("Print partition statistics");

    const auto cut       = metrics::edge_cut(p_graph);
    const auto imbalance = metrics::imbalance(p_graph);
    const bool feasible  = metrics::is_feasible(p_graph, p_ctx);

    LOG << "Initial partition:";
    LOG << "  Number of blocks: " << p_graph.k();
    LOG << "  Cut:              " << cut;
    LOG << "  Imbalance:        " << imbalance;
    LOG << "  Feasible:         " << (feasible ? "yes" : "no");
}

DistributedPartitionedGraph DeepPartitioningScheme::partition() {
    // const PEID size = mpi::get_comm_size(_input_graph.communicator());

    const DistributedGraph* graph     = &_input_graph;
    auto*                   coarsener = get_current_coarsener();
    bool                    converged = false;

    /*
     * Coarsening
     */
    const BlockID      first_step_k = std::min<BlockID>(_input_ctx.partition.k, _input_ctx.partition.K);
    const GlobalNodeID desired_num_nodes =
        _input_ctx.parallel.num_threads * _input_ctx.coarsening.contraction_limit * first_step_k;
    while (!converged && graph->global_n() > desired_num_nodes) {
        SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(coarsener->level()));

        const GlobalNodeWeight  max_cluster_weight = coarsener->max_cluster_weight();
        const DistributedGraph* c_graph            = coarsener->coarsen_once();
        converged                                  = (graph == c_graph);

        if (!converged) {
            print_coarsening_level(max_cluster_weight);
            if (c_graph->global_n() <= desired_num_nodes) {
                print_coarsening_terminated(desired_num_nodes);
            }
        } else {
            print_coarsening_converged();
        }

        graph = c_graph;
    }

    /*
     * Initial Partitioning
     */
    START_TIMER("Initial partitioning");
    auto initial_partitioner = TIMED_SCOPE("Allocation") {
        return factory::create_initial_partitioning_algorithm(_input_ctx);
    };

    auto shm_graph = graph::replicate_everywhere(*graph);
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && _input_ctx.debug.save_coarsest_graph) {
        debug::save_graph(shm_graph, _input_ctx, coarsener->level());
    }

    PartitionContext ip_p_ctx = _input_ctx.partition;
    ip_p_ctx.k                = first_step_k;
    ip_p_ctx.epsilon          = _input_ctx.partition.epsilon;
    ip_p_ctx.setup(shm_graph);
    auto shm_p_graph = initial_partitioner->initial_partition(shm_graph, ip_p_ctx);

    DistributedPartitionedGraph dist_p_graph = graph::distribute_best_partition(*graph, std::move(shm_p_graph));

    KASSERT(graph::debug::validate_partition(dist_p_graph), "", assert::heavy);

    print_initial_partitioning_result(dist_p_graph, ip_p_ctx);
    STOP_TIMER();

    /*
     * Uncoarsening and Refinement
     */
    START_TIMER("Uncoarsening");
    auto refinement_algorithm = TIMED_SCOPE("Allocation") {
        return factory::create_refinement_algorithm(_input_ctx);
    };

    auto run_balancer = [&](DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
        SCOPED_TIMER("Rebalancing");
        if (!metrics::is_feasible(p_graph, p_ctx)) {
            DistributedBalancer balancer(_input_ctx);
            balancer.initialize(p_graph);
            balancer.balance(p_graph, p_ctx);
            KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);
        }
    };

    auto run_local_search = [&](DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
        SCOPED_TIMER("Local search");
        refinement_algorithm->initialize(p_graph.graph(), p_ctx);
        refinement_algorithm->refine(p_graph);
        KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);
    };

    auto extend_partition = [&](DistributedPartitionedGraph& p_graph) {
        START_TIMER("Extending partition", std::string("Level ") + std::to_string(coarsener->level()));
        BlockID desired_k = std::min<BlockID>(
            _input_ctx.partition.k, math::ceil2(dist_p_graph.global_n() / _input_ctx.coarsening.contraction_limit)
        );
        if (_input_graph.global_n() == p_graph.global_n()) {
            // If we work on the input graph, extend to final number of blocks
            desired_k = _input_ctx.partition.k;
        }
        while (dist_p_graph.k() < desired_k) {
            const BlockID next_k = std::min<BlockID>(desired_k, dist_p_graph.k() * _input_ctx.partition.K);
            KASSERT(next_k % dist_p_graph.k() == 0u);
            const BlockID k_per_block = next_k / dist_p_graph.k();

            LOG << "  Extending partition from " << dist_p_graph.k() << " blocks to " << next_k << " blocks";

            // Extract blocks
            auto        block_extraction_result = graph::extract_and_scatter_block_induced_subgraphs(dist_p_graph);
            const auto& subgraphs               = block_extraction_result.subgraphs;

            // Partition block-induced subgraphs

            START_TIMER("Initial partitioning");
            std::vector<shm::PartitionedGraph> p_subgraphs;
            for (const auto& subgraph: subgraphs) {
                const double target_max_block_weight =
                    (1.0 + _input_ctx.partition.epsilon) * _input_graph.global_total_node_weight() / next_k;
                const double next_epsilon =
                    1.0 * target_max_block_weight / subgraph.total_node_weight() * k_per_block - 1.0;
                ip_p_ctx.k       = k_per_block;
                ip_p_ctx.epsilon = std::max(0.001, next_epsilon);

                ip_p_ctx.setup(subgraph);
                p_subgraphs.push_back(initial_partitioner->initial_partition(subgraph, ip_p_ctx));

                DBG << "Next subgraph with epsilon=" << next_epsilon
                    << " and total_node_weight=" << subgraph.total_node_weight()
                    << " | target_max_block_weight=" << target_max_block_weight << " | k_per_block=" << k_per_block
                    << " | next_k=" << next_k;
            }
            STOP_TIMER();

            // Project subgraph partitions onto dist_p_graph
            dist_p_graph =
                graph::copy_subgraph_partitions(std::move(dist_p_graph), p_subgraphs, block_extraction_result);

            // Print statistics
            START_TIMER("Print partition statistics");
            const auto cut       = metrics::edge_cut(dist_p_graph);
            const auto imbalance = metrics::imbalance(dist_p_graph);

            ip_p_ctx.k       = next_k;
            ip_p_ctx.epsilon = _input_ctx.partition.epsilon;
            ip_p_ctx.setup(dist_p_graph.graph());
            const bool feasible = metrics::is_feasible(dist_p_graph, ip_p_ctx);

            LOG << "    Cut:       " << cut;
            LOG << "    Imbalance: " << std::setprecision(3) << imbalance;
            LOG << "    Feasible:  " << (feasible ? "yes" : "no");
            STOP_TIMER();
        }
        STOP_TIMER();
    };

    auto run_refinement = [&](DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
        run_balancer(p_graph, p_ctx);
        run_local_search(p_graph, p_ctx);
        run_balancer(p_graph, p_ctx);
    };

    auto ref_p_ctx = _input_ctx.partition;
    ref_p_ctx.setup(dist_p_graph.graph());

    // Uncoarsen, partition blocks and refine
    while (coarsener->level() > 0) {
        LOG;
        LOG << "Uncoarsening -> Level " << coarsener->level() << ":";

        // Uncoarsen graph
        START_TIMER("Uncontraction", std::string("Level ") + std::to_string(coarsener->level()));
        dist_p_graph = coarsener->uncoarsen_once(std::move(dist_p_graph));
        STOP_TIMER();

        // Extend partition
        extend_partition(dist_p_graph);

        // Run refinement
        START_TIMER("Refinement", std::string("Level ") + std::to_string(coarsener->level()));
        LOG << "  Running balancing and local search on " << dist_p_graph.k() << " blocks";
        ref_p_ctx.k       = dist_p_graph.k();
        ref_p_ctx.epsilon = _input_ctx.partition.epsilon;
        ref_p_ctx.setup(dist_p_graph.graph());
        run_refinement(dist_p_graph, ref_p_ctx);

        // Output statistics
        START_TIMER("Print partition statistics");
        const auto cut       = metrics::edge_cut(dist_p_graph);
        const auto imbalance = metrics::imbalance(dist_p_graph);
        const bool feasible  = metrics::is_feasible(dist_p_graph, ref_p_ctx);
        LOG << "  Cut:       " << cut;
        LOG << "  Imbalance: " << imbalance;
        LOG << "  Feasible:  " << (feasible ? "yes" : "no");
        STOP_TIMER();
        STOP_TIMER();
    }

    // Extend partition if we have not already reached the desired number of blocks
    // This should only be used to cover the special case where the input graph is too small for coarsening
    if (dist_p_graph.k() != _input_ctx.partition.k) {
        LOG;
        LOG << "Flat partitioning:";

        // Extend partition
        extend_partition(dist_p_graph);

        // Run refinement
        START_TIMER("Refinement", std::string("Level ") + std::to_string(coarsener->level()));

        LOG << "  Running balancing and local search on " << dist_p_graph.k() << " blocks";
        ref_p_ctx.k       = dist_p_graph.k();
        ref_p_ctx.epsilon = _input_ctx.partition.epsilon;
        ref_p_ctx.setup(dist_p_graph.graph());
        run_refinement(dist_p_graph, ref_p_ctx);

        // Output statistics
        START_TIMER("Print partition statistics");
        const auto cut       = metrics::edge_cut(dist_p_graph);
        const auto imbalance = metrics::imbalance(dist_p_graph);
        const bool feasible  = metrics::is_feasible(dist_p_graph, ref_p_ctx);
        LOG << "  Cut:       " << cut;
        LOG << "  Imbalance: " << imbalance;
        LOG << "  Feasible:  " << (feasible ? "yes" : "no");
        STOP_TIMER();
        STOP_TIMER();
    }
    STOP_TIMER();

    return dist_p_graph;
}

Coarsener* DeepPartitioningScheme::get_current_coarsener() {
    KASSERT(!_coarseners.empty());
    return &_coarseners.top();
}

const Coarsener* DeepPartitioningScheme::get_current_coarsener() const {
    KASSERT(!_coarseners.empty());
    return &_coarseners.top();
}
} // namespace kaminpar::dist
