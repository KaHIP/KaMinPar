/*******************************************************************************
 * @file:   coarsener.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Builds and manages a hierarchy of coarse graphs.
 ******************************************************************************/
#include "dkaminpar/coarsening/coarsener.h"

#include "context.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/factories.h"

namespace dkaminpar {
Coarsener::Coarsener(const DistributedGraph& input_graph, const Context& input_ctx)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {
    _clustering_algorithm = factory::create_clustering_algorithm(_input_ctx);
}

const DistributedGraph* Coarsener::coarsen_once() {
    return coarsen_once(max_cluster_weight());
}

const DistributedGraph* Coarsener::coarsen_once(const GlobalNodeWeight max_cluster_weight) {
    const DistributedGraph* graph = coarsest();

    // compute coarse graph
    auto& clustering = _clustering_algorithm->compute_clustering(*graph, static_cast<NodeWeight>(max_cluster_weight));
    auto [c_graph, mapping] =
        coarsening::contract_global_clustering(*graph, clustering, _input_ctx.coarsening.global_contraction_algorithm);
    HEAVY_ASSERT(graph::debug::validate(c_graph));

    // only keep graph if coarsening has not converged yet
    const bool converged = (1.0 * c_graph.global_n() / graph->global_n()) >= 0.95;
    if (!converged) {
        _graph_hierarchy.push_back(std::move(c_graph));
        _mapping_hierarchy.push_back(std::move(mapping));
        return coarsest();
    }

    return graph;
}

DistributedPartitionedGraph Coarsener::uncoarsen_once(DistributedPartitionedGraph&& p_graph) {
    ASSERT(coarsest() == &p_graph.graph()) << "expected graph partition of current coarsest graph";

    const DistributedGraph* new_coarsest = nth_coarsest(1);
    p_graph = coarsening::project_global_contracted_graph(*new_coarsest, std::move(p_graph), _mapping_hierarchy.back());
    HEAVY_ASSERT(graph::debug::validate_partition(p_graph));

    _graph_hierarchy.pop_back();
    _mapping_hierarchy.pop_back();

    // if pop_back() on _graph_hierarchy caused a reallocation, the graph pointer in p_graph dangles
    p_graph.UNSAFE_set_graph(coarsest());

    return std::move(p_graph);
}

const DistributedGraph* Coarsener::coarsest() const {
    return nth_coarsest(0);
}

std::size_t Coarsener::level() const {
    return _graph_hierarchy.size();
}

const DistributedGraph* Coarsener::nth_coarsest(const std::size_t n) const {
    return _graph_hierarchy.size() > n ? &_graph_hierarchy[_graph_hierarchy.size() - n - 1] : &_input_graph;
}

GlobalNodeWeight Coarsener::max_cluster_weight() const {
    shm::PartitionContext shm_p_ctx = _input_ctx.initial_partitioning.sequential.partition;
    shm_p_ctx.k                     = _input_ctx.partition.k;
    shm_p_ctx.epsilon               = _input_ctx.partition.epsilon;

    shm::CoarseningContext shm_c_ctx    = _input_ctx.initial_partitioning.sequential.coarsening;
    shm_c_ctx.contraction_limit         = _input_ctx.coarsening.contraction_limit;
    shm_c_ctx.cluster_weight_limit      = _input_ctx.coarsening.cluster_weight_limit;
    shm_c_ctx.cluster_weight_multiplier = _input_ctx.coarsening.cluster_weight_multiplier;

    const auto* graph = coarsest();
    return shm::compute_max_cluster_weight<GlobalNodeID, GlobalNodeWeight>(
        graph->global_n(), graph->global_total_node_weight(), shm_p_ctx, shm_c_ctx);
}
} // namespace dkaminpar
