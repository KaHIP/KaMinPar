/*******************************************************************************
 * @file:   kway.cc
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Direct k-way partitioning.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/kway.h"

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/utils/metrics.h"
#include "kaminpar/metrics.h"
#include "kaminpar/utils/console_io.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar {
SET_DEBUG(true);

KWayPartitioningScheme::KWayPartitioningScheme(const DistributedGraph& graph, const Context& ctx)
    : _graph{graph},
      _ctx{ctx} {}

DistributedPartitionedGraph KWayPartitioningScheme::partition() {
    // Coarsen graph
    std::vector<DistributedGraph>          graph_hierarchy;
    std::vector<coarsening::GlobalMapping> mapping_hierarchy;

    const DistributedGraph* c_graph = &_graph;

    ////////////////////////////////////////////////////////////////////////////////
    // Step 1: Coarsening
    ////////////////////////////////////////////////////////////////////////////////
    shm::cio::print_banner("Coarsening");

    {
        auto clustering_algorithm = TIMED_SCOPE("Allocation") {
            return factory::create_global_clustering(_ctx);
        };

        while (c_graph->global_n() > _ctx.partition.k * _ctx.coarsening.contraction_limit) {
            SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(graph_hierarchy.size()));

            shm::PartitionContext shm_p_ctx = _ctx.initial_partitioning.sequential.partition;
            shm_p_ctx.k                     = _ctx.partition.k;
            shm_p_ctx.epsilon               = _ctx.partition.epsilon;

            shm::CoarseningContext shm_c_ctx    = _ctx.initial_partitioning.sequential.coarsening;
            shm_c_ctx.contraction_limit         = _ctx.coarsening.contraction_limit;
            shm_c_ctx.cluster_weight_limit      = _ctx.coarsening.cluster_weight_limit;
            shm_c_ctx.cluster_weight_multiplier = _ctx.coarsening.cluster_weight_multiplier;

            const GlobalNodeWeight max_cluster_weight = shm::compute_max_cluster_weight<GlobalNodeID, GlobalNodeWeight>(
                c_graph->global_n(), c_graph->global_total_node_weight(), shm_p_ctx, shm_c_ctx);

            ALWAYS_ASSERT(max_cluster_weight <= std::numeric_limits<NodeWeight>::max());
            auto& clustering =
                clustering_algorithm->compute_clustering(*c_graph, static_cast<NodeWeight>(max_cluster_weight));

            auto [contracted_graph, mapping] = coarsening::contract_global_clustering(
                *c_graph, clustering, _ctx.coarsening.global_contraction_algorithm);
            HEAVY_ASSERT(graph::debug::validate(contracted_graph));

            const bool converged = contracted_graph.global_n() == c_graph->global_n();
            graph_hierarchy.push_back(std::move(contracted_graph));
            mapping_hierarchy.push_back(std::move(mapping));
            c_graph = &graph_hierarchy.back();

            // Print statistics for coarse graph
            const std::string n_str       = mpi::gather_statistics_str(c_graph->n(), c_graph->communicator());
            const std::string ghost_n_str = mpi::gather_statistics_str(c_graph->ghost_n(), c_graph->communicator());
            const std::string m_str       = mpi::gather_statistics_str(c_graph->m(), c_graph->communicator());
            const std::string max_node_weight_str =
                mpi::gather_statistics_str<GlobalNodeWeight>(c_graph->max_node_weight(), c_graph->communicator());

            LOG << "=> level=" << graph_hierarchy.size() << " "
                << "global_n=" << c_graph->global_n() << " "
                << "global_m=" << c_graph->global_m() << " "
                << "n=[" << n_str << "] "
                << "ghost_n=[" << ghost_n_str << "] "
                << "m=[" << m_str << "] "
                << "max_node_weight=[" << max_node_weight_str << "] "
                << "max_cluster_weight=" << max_cluster_weight;

            if (converged) {
                LOG << "==> Coarsening converged";
                break;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Step 2: Initial Partitioning
    ////////////////////////////////////////////////////////////////////////////////
    shm::cio::print_banner("Initial Partitioning");

    auto initial_partitioner = TIMED_SCOPE("Allocation") {
        return factory::create_initial_partitioner(_ctx);
    };

    START_TIMER("Initial Partitioning");
    auto                        shm_graph    = graph::allgather(*c_graph);
    auto                        shm_p_graph  = initial_partitioner->initial_partition(shm_graph);
    DistributedPartitionedGraph dist_p_graph = graph::reduce_scatter(*c_graph, std::move(shm_p_graph));
    STOP_TIMER();

    HEAVY_ASSERT(graph::debug::validate_partition(dist_p_graph));

    const auto initial_cut       = metrics::edge_cut(dist_p_graph);
    const auto initial_imbalance = metrics::imbalance(dist_p_graph);

    LOG << "Initial partition: cut=" << initial_cut << " imbalance=" << initial_imbalance;

    ////////////////////////////////////////////////////////////////////////////////
    // Step 3: Refinement
    ////////////////////////////////////////////////////////////////////////////////
    shm::cio::print_banner("Refinement");

    auto refinement_algorithm = TIMED_SCOPE("Allocation") {
        return factory::create_distributed_refiner(_ctx);
    };

    auto refine = [&](DistributedPartitionedGraph& p_graph) {
        SCOPED_TIMER("Refinement");
        refinement_algorithm->initialize(p_graph.graph(), _ctx.partition);
        refinement_algorithm->refine(p_graph);
        HEAVY_ASSERT(graph::debug::validate_partition(p_graph));
    };

    // Uncoarsen and refine
    while (!graph_hierarchy.empty()) {
        SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(graph_hierarchy.size()));

        {
            SCOPED_TIMER("Uncontraction");

            const auto* current_graph =
                graph_hierarchy.size() <= 1 ? &_graph : &graph_hierarchy[graph_hierarchy.size() - 2];
            HEAVY_ASSERT(graph::debug::validate(*current_graph));

            dist_p_graph = coarsening::project_global_contracted_graph(
                *current_graph, std::move(dist_p_graph), mapping_hierarchy.back());
            HEAVY_ASSERT(graph::debug::validate_partition(dist_p_graph));

            graph_hierarchy.pop_back();
            mapping_hierarchy.pop_back();

            // update graph ptr in case graph_hierarchy was reallocated by the pop_back() operation
            dist_p_graph.UNSAFE_set_graph(graph_hierarchy.empty() ? &_graph : &graph_hierarchy.back());
        }

        refine(dist_p_graph);

        // Output refinement statistics
        const auto current_cut       = metrics::edge_cut(dist_p_graph);
        const auto current_imbalance = metrics::imbalance(dist_p_graph);

        LOG << "=> level=" << graph_hierarchy.size() << " cut=" << current_cut << " imbalance=" << current_imbalance;
    }

    return dist_p_graph;
}
} // namespace dkaminpar
