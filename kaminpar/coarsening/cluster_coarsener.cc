/*******************************************************************************
 * @file:   cluster_coarsener.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Coarsener that uses a clustering algorithm to coarsen the graph.
 ******************************************************************************/
#include "kaminpar/coarsening/cluster_coarsener.h"

#include "kaminpar/utility/timer.h"

namespace kaminpar {
std::pair<const Graph *, bool> ClusteringCoarsener::compute_coarse_graph(const NodeWeight max_cluster_weight,
                                                                         const NodeID to_size) {
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()), TIMER_BENCHMARK);

  _clustering_algorithm->set_max_cluster_weight(max_cluster_weight);
  _clustering_algorithm->set_desired_cluster_count(to_size);

  const auto &clustering = TIMED_SCOPE("Label Propagation") {
    return _clustering_algorithm->compute_clustering(*_current_graph);
  };

  auto [c_graph, c_mapping, m_ctx] = TIMED_SCOPE("Contract graph") {
    return graph::contract(*_current_graph, clustering, std::move(_contraction_m_ctx));
  };
  _contraction_m_ctx = std::move(m_ctx);

  const bool converged = _c_ctx.should_converge(_current_graph->n(), c_graph.n());

  _hierarchy.push_back(std::move(c_graph));
  _mapping.push_back(std::move(c_mapping));
  _current_graph = &_hierarchy.back();

  return {_current_graph, !converged};
}

PartitionedGraph ClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  ASSERT(&p_graph.graph() == _current_graph);
  ASSERT(!empty()) << size();
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()), TIMER_BENCHMARK);

  START_TIMER("Allocation");
  auto mapping{std::move(_mapping.back())};
  _mapping.pop_back();
  _hierarchy.pop_back(); // destroys the graph wrapped in p_graph, but partition access is still ok
  _current_graph = empty() ? &_input_graph : &_hierarchy.back();
  ASSERT(mapping.size() == _current_graph->n()) << V(mapping.size()) << V(_current_graph->n());

  StaticArray<parallel::IntegralAtomicWrapper<BlockID>> partition(_current_graph->n());
  STOP_TIMER();

  START_TIMER("Copy partition");
  tbb::parallel_for(static_cast<NodeID>(0), _current_graph->n(),
                    [&](const NodeID u) { partition[u] = p_graph.block(mapping[u]); });
  STOP_TIMER();

  SCOPED_TIMER("Create graph");
  PartitionedGraph new_p_graph(*_current_graph, p_graph.k(), std::move(partition), std::move(p_graph.take_final_k()));
#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
  new_p_graph.set_block_names(p_graph.block_names());
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES

  return new_p_graph;
}
} // namespace kaminpar