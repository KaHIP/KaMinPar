/*******************************************************************************
 * Graph coarsener based on global clusterings.
 *
 * @file:   global_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#include "kaminpar-dist/coarsening/global_cluster_coarsener.h"

#include "kaminpar-dist/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(false);
}

GlobalClusterCoarsener::GlobalClusterCoarsener(const Context &input_ctx)
    : _input_ctx(input_ctx),
      _clusterer(factory::create_global_clusterer(_input_ctx)) {}

void GlobalClusterCoarsener::initialize(const DistributedGraph *graph) {
  _input_graph = graph;
  _graph_hierarchy.clear();
}

bool GlobalClusterCoarsener::coarsen() {
  DBG << "Coarsen graph using global clustering algorithm ...";

  const DistributedGraph &graph = current();

  StaticArray<GlobalNodeID> clustering(graph.total_n(), static_array::noinit);

  _clusterer->set_max_cluster_weight(max_cluster_weight());
  _clusterer->cluster(clustering, graph);

  auto result = contract_clustering(graph, clustering, _input_ctx.coarsening);

  KASSERT(
      debug::validate_graph(result.graph),
      "invalid graph after global cluster contraction",
      assert::heavy
  );
  DBG << "Reduced number of nodes from " << graph.global_n() << " to " << result.graph.global_n();

  if (!has_converged(graph, result.graph)) {
    DBG << "... success";

    _graph_hierarchy.push_back(std::move(result.graph));
    _global_mapping_hierarchy.push_back(std::move(result.mapping));
    _node_migration_history.push_back(std::move(result.migration));

    return true;
  }

  DBG << "... converged due to insufficient shrinkage";
  return false;
}

DistributedPartitionedGraph GlobalClusterCoarsener::uncoarsen(DistributedPartitionedGraph &&p_graph
) {
  const DistributedGraph *new_coarsest = nth_coarsest(1);

  p_graph = project_partition(
      *new_coarsest,
      std::move(p_graph),
      _global_mapping_hierarchy.back(),
      _node_migration_history.back()
  );

  KASSERT(
      debug::validate_partition(p_graph),
      "invalid partition after projection to finer graph",
      assert::heavy
  );

  _graph_hierarchy.pop_back();
  _global_mapping_hierarchy.pop_back();
  _node_migration_history.pop_back();

  // if pop_back() on _graph_hierarchy caused a reallocation, the graph pointer
  // in p_graph dangles
  p_graph.UNSAFE_set_graph(coarsest());

  return std::move(p_graph);
}

bool GlobalClusterCoarsener::has_converged(
    const DistributedGraph &before, const DistributedGraph &after
) const {
  return 1.0 * after.global_n() / before.global_n() >= 0.95;
}

const DistributedGraph &GlobalClusterCoarsener::current() const {
  return _graph_hierarchy.empty() ? *_input_graph : _graph_hierarchy.back();
}

std::size_t GlobalClusterCoarsener::level() const {
  return _graph_hierarchy.size();
}

GlobalNodeWeight GlobalClusterCoarsener::max_cluster_weight() const {
  return shm::compute_max_cluster_weight<GlobalNodeWeight>(
      _input_ctx.coarsening,
      _input_ctx.partition,
      current().global_n(),
      current().global_total_node_weight()
  );
}
} // namespace kaminpar::dist

