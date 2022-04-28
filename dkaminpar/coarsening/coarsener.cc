/*******************************************************************************
 * @file:   coarsener.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Builds and manages a hierarchy of coarse graphs.
 ******************************************************************************/
#include "dkaminpar/coarsening/coarsener.h"

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
    HEAVY_ASSERT(graph::debug::validate(contracted_graph));

    // only keep graph if coarsening has not converged yet
    const bool converged = (c_graph.global_n() == graph->global_n());
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
    return _graph_hierarchy.size() > n ? &_graph_hierarchy[_graph_hierarchy.size() - n] : &_input_graph;
}

GlobalNodeWeight Coarsener::max_cluster_weight() const {
    return 0; // @todo
}
} // namespace dkaminpar
