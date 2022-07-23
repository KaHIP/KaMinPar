/*******************************************************************************
 * @file:   deep_partitioning_scheme.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Deep multilevel graph partitioning scheme.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/deep_partitioning_scheme.h"

#include <mpi.h>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/partitioning_scheme/kway_partitioning_scheme.h"

#include "kaminpar/utils/timer.h"

#include "common/utils/math.h"

namespace dkaminpar {
DeepPartitioningScheme::DeepPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {
    _coarseners.emplace(_input_graph, _input_ctx);
}

DistributedPartitionedGraph DeepPartitioningScheme::partition() {
    // repeat coarsening until no. of vertices <= C * P
    // duplicate s.t. each MPI process has >= C vertices
    // ASSERTION: if graph has C vertices, each MPI process has its own copy
    // partition graph into C / C' blocks, for some C'
    // uncontract, join duplicates, take better partition
    // --> until each block has >= C nodes
    //  --> distribute blocks to PEs, partition
    const int               initial_comm_size   = mpi::get_comm_size(_input_graph.communicator());
    int                     current_comm_size   = initial_comm_size;
    const int               initial_comm_rank   = mpi::get_comm_rank(_input_graph.communicator());
    int                     current_comm_rank   = initial_comm_rank;
    const int               initial_parallelism = _input_ctx.parallel.num_threads * initial_comm_size;
    int                     current_parallelism = initial_parallelism;
    const DistributedGraph* current_graph       = &_input_graph;

    auto*                        current_coarsener = get_current_coarsener();
    bool                         converged         = false;
    std::stack<DistributedGraph> gathered_graphs;

    while (!converged && current_graph->global_n() > _input_ctx.coarsening.contraction_limit) {
        while (!converged
               && current_graph->global_n() >= current_parallelism * _input_ctx.coarsening.contraction_limit) {
            SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(current_coarsener->level()));

            const GlobalNodeWeight max_cluster_weight = current_coarsener->max_cluster_weight();

            const DistributedGraph* c_graph = current_coarsener->coarsen_once();
            converged                       = (current_graph == c_graph);

            if (!converged) {
                // Print statistics for coarse graph
                const std::string n_str       = mpi::gather_statistics_str(c_graph->n(), c_graph->communicator());
                const std::string ghost_n_str = mpi::gather_statistics_str(c_graph->ghost_n(), c_graph->communicator());
                const std::string m_str       = mpi::gather_statistics_str(c_graph->m(), c_graph->communicator());
                const std::string max_node_weight_str =
                    mpi::gather_statistics_str<GlobalNodeWeight>(c_graph->max_node_weight(), c_graph->communicator());

                // Machine readable
                LOG << "=> level=" << current_coarsener->level() << " "
                    << "global_n=" << c_graph->global_n() << " "
                    << "global_m=" << c_graph->global_m() << " "
                    << "n=[" << n_str << "] "
                    << "ghost_n=[" << ghost_n_str << "] "
                    << "m=[" << m_str << "] "
                    << "max_node_weight=[" << max_node_weight_str << "] "
                    << "max_cluster_weight=" << max_cluster_weight;

                // Human readable
                LOG << "Level " << current_coarsener->level() << ":";
                graph::print_summary(*c_graph);

                current_graph = c_graph;
            }
        }

        if (!converged) {
            // split communicator into groups
            const unsigned int desired_multiplicity =
                1.0 * _input_ctx.coarsening.contraction_limit * current_parallelism / current_graph->global_n();
            const unsigned int multiplicity = std::min<int>(current_comm_size, shm::math::ceil2(desired_multiplicity));
            const int          new_size     = current_comm_size / multiplicity;
            const int          new_group    = current_comm_rank / new_size;

            MPI_Comm new_comm;
            MPI_Comm_split(current_graph->communicator(), new_group, current_comm_rank, &new_comm);

            // duplicate graph s.t. each new communicator group owns a full copy of the current graph
            gathered_graphs.push(graph::allgather_on_groups(*current_graph, new_comm));

            current_graph       = &gathered_graphs.top();
            current_parallelism = _input_ctx.parallel.num_threads * new_size;

            _coarseners.emplace(*current_graph, _input_ctx);
            current_coarsener = get_current_coarsener();
        }
    }

    if (converged) {
        LOG << "==> Coarsening converged";
    }

    return KWayPartitioningScheme(_input_graph, _input_ctx).partition();
}

Coarsener* DeepPartitioningScheme::get_current_coarsener() {
    KASSERT(!_coarseners.empty());
    return &_coarseners.top();
}
} // namespace dkaminpar
