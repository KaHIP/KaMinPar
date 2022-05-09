/*******************************************************************************
 * @file:   kway.cc
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Partitioning scheme using direct k-way partitioning.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/kway.h"

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/refinement/distributed_balancer.h"
#include "dkaminpar/utils/metrics.h"
#include "kaminpar/metrics.h"
#include "kaminpar/utils/console_io.h"
#include "kaminpar/utils/strings.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar {
SET_DEBUG(false);

namespace {
void save_imbalanced_graph_partition(const DistributedPartitionedGraph& p_graph, const Context& ctx, int level) {
    const std::string base = shm::utility::str::extract_basename(ctx.graph_filename) + ".seed"
                             + std::to_string(ctx.seed) + ".k" + std::to_string(ctx.partition.k) + ".level"
                             + std::to_string(level);
    const std::string graph_filename     = base + ".graph";
    const std::string partition_filename = base + ".part";

    io::metis::write(graph_filename, p_graph.graph());

    std::vector<BlockID> partition(p_graph.n());
    std::copy_n(p_graph.partition().begin(), p_graph.n(), partition.begin());
    io::partition::write(partition_filename, partition);
}
} // namespace

KWayPartitioningScheme::KWayPartitioningScheme(const DistributedGraph& graph, const Context& ctx)
    : _graph{graph},
      _ctx{ctx} {}

DistributedPartitionedGraph KWayPartitioningScheme::partition() {
    Coarsener coarsener(_graph, _ctx);

    const DistributedGraph* graph = &_graph;

    ////////////////////////////////////////////////////////////////////////////////
    // Step 1: Coarsening
    ////////////////////////////////////////////////////////////////////////////////
    if (mpi::get_comm_rank() == 0) {
        shm::cio::print_banner("Coarsening");
    }

    {
        SCOPED_TIMER("Coarsening");

        while (graph->global_n() > _ctx.partition.k * _ctx.coarsening.contraction_limit) {
            SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(coarsener.level()));
            const GlobalNodeWeight max_cluster_weight = coarsener.max_cluster_weight();

            const DistributedGraph* c_graph   = coarsener.coarsen_once();
            const bool              converged = (graph == c_graph);

            if (!converged) {
                // Print statistics for coarse graph
                const std::string n_str       = mpi::gather_statistics_str(c_graph->n(), c_graph->communicator());
                const std::string ghost_n_str = mpi::gather_statistics_str(c_graph->ghost_n(), c_graph->communicator());
                const std::string m_str       = mpi::gather_statistics_str(c_graph->m(), c_graph->communicator());
                const std::string max_node_weight_str =
                    mpi::gather_statistics_str<GlobalNodeWeight>(c_graph->max_node_weight(), c_graph->communicator());

                LOG << "=> level=" << coarsener.level() << " "
                    << "global_n=" << c_graph->global_n() << " "
                    << "global_m=" << c_graph->global_m() << " "
                    << "n=[" << n_str << "] "
                    << "ghost_n=[" << ghost_n_str << "] "
                    << "m=[" << m_str << "] "
                    << "max_node_weight=[" << max_node_weight_str << "] "
                    << "max_cluster_weight=" << max_cluster_weight;

                graph = c_graph;
            } else if (converged) {
                LOG << "==> Coarsening converged";
                break;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Step 2: Initial Partitioning
    ////////////////////////////////////////////////////////////////////////////////
    if (mpi::get_comm_rank() == 0) {
        shm::cio::print_banner("Initial Partitioning");
    }

    auto initial_partitioner = TIMED_SCOPE("Allocation") {
        return factory::create_initial_partitioner(_ctx);
    };

    START_TIMER("Initial Partitioning");
    auto                        shm_graph    = graph::allgather(*graph);
    auto                        shm_p_graph  = initial_partitioner->initial_partition(shm_graph);
    DistributedPartitionedGraph dist_p_graph = graph::reduce_scatter(*graph, std::move(shm_p_graph));
    STOP_TIMER();

    KASSERT(graph::debug::validate_partition(dist_p_graph), "", assert::heavy);

    const auto initial_cut       = metrics::edge_cut(dist_p_graph);
    const auto initial_imbalance = metrics::imbalance(dist_p_graph);

    LOG << "Initial partition: cut=" << initial_cut << " imbalance=" << initial_imbalance;

    if (_ctx.save_imbalanced_partitions && initial_imbalance > _ctx.partition.epsilon) {
        save_imbalanced_graph_partition(dist_p_graph, _ctx, static_cast<int>(coarsener.level()));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Step 3: Refinement
    ////////////////////////////////////////////////////////////////////////////////
    {
        SCOPED_TIMER("Uncoarsening");

        if (mpi::get_comm_rank() == 0) {
            shm::cio::print_banner("Refinement");
        }

        auto refinement_algorithm = TIMED_SCOPE("Allocation") {
            return factory::create_distributed_refiner(_ctx);
        };

        auto refine = [&](DistributedPartitionedGraph& p_graph) {
            {
                SCOPED_TIMER("Balancing");
                if (metrics::imbalance(p_graph) > _ctx.partition.epsilon) {
                    DistributedBalancer balancer(_ctx);
                    balancer.initialize(p_graph);
                    balancer.balance(p_graph, _ctx.partition);
                }
            }
            {
                SCOPED_TIMER("Refinement");
                refinement_algorithm->initialize(p_graph.graph(), _ctx.partition);
                refinement_algorithm->refine(p_graph);
                KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);
            }
        };

        // Uncoarsen and refine
        while (coarsener.level() > 0) {
            SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener.level()));

            dist_p_graph = TIMED_SCOPE("Uncontraction") {
                return coarsener.uncoarsen_once(std::move(dist_p_graph));
            };

            refine(dist_p_graph);

            // Output refinement statistics
            const auto current_cut       = metrics::edge_cut(dist_p_graph);
            const auto current_imbalance = metrics::imbalance(dist_p_graph);

            LOG << "=> level=" << coarsener.level() << " cut=" << current_cut << " imbalance=" << current_imbalance;

            if (_ctx.save_imbalanced_partitions && current_imbalance > _ctx.partition.epsilon) {
                save_imbalanced_graph_partition(dist_p_graph, _ctx, static_cast<int>(coarsener.level()));
            }
        }
    }

    return dist_p_graph;
}
} // namespace dkaminpar
