/*******************************************************************************
 * Builds and manages a hierarchy of coarse graphs.
 *
 * @file:   coarsener.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#include "kaminpar-dist/coarsening/coarsener.h"

#include "kaminpar-dist/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-dist/coarsening/contraction/local_cluster_contraction.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/factories.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/partition_utils.h"

namespace kaminpar::dist {
SET_DEBUG(false);

Coarsener::Coarsener(const DistributedGraph &input_graph, const Context &input_ctx)
    : _input_graph(input_graph),
      _input_ctx(input_ctx),
      _global_clusterer(factory::create_global_clusterer(_input_ctx)),
      _local_clusterer(factory::create_local_clusterer(_input_ctx)) {}

const DistributedGraph *Coarsener::coarsen_once() {
  return coarsen_once(max_cluster_weight());
}

const DistributedGraph *Coarsener::coarsen_once_local(const GlobalNodeWeight max_cluster_weight) {
  DBG << "Coarsen graph using local clustering algorithm ...";
  const DistributedGraph *graph = coarsest();

  _local_clusterer->initialize(*graph);
  auto &clustering = _local_clusterer->cluster(*graph, max_cluster_weight);
  if (clustering.empty()) {
    DBG << "... converged with empty clustering";
    return graph;
  }

  scalable_vector<parallel::Atomic<NodeID>> legacy_clustering(clustering.begin(), clustering.end());
  auto [c_graph, mapping, m_ctx] = contract_local_clustering(*graph, legacy_clustering);
  KASSERT(debug::validate_graph(c_graph), "", assert::heavy);
  DBG << "Reduced number of nodes from " << graph->global_n() << " to " << c_graph.global_n();

  if (!has_converged(*graph, c_graph)) {
    DBG << "... success";

    _graph_hierarchy.push_back(std::move(c_graph));
    _local_mapping_hierarchy.push_back(std::move(mapping));
    return coarsest();
  }

  DBG << "... converged due to insufficient shrinkage";
  return graph;
}

const DistributedGraph *Coarsener::coarsen_once_global(const GlobalNodeWeight max_cluster_weight) {
  DBG << "Coarsen graph using global clustering algorithm ...";

  const DistributedGraph *graph = coarsest();

  // Call global clustering algorithm
  _global_clusterer->initialize(*graph);
  auto &clustering =
      _global_clusterer->cluster(*graph, static_cast<NodeWeight>(max_cluster_weight));

  bool empty = clustering.empty();
  MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPI_CXX_BOOL, MPI_LAND, graph->communicator());
  if (empty) { // Empty --> converged
    DBG << "... converged with empty clustering";
    return graph;
  }

  // Construct the coarse graph
  auto result = contract_clustering(*graph, clustering, _input_ctx.coarsening);

  KASSERT(debug::validate_graph(result.graph), "", assert::heavy);
  DBG << "Reduced number of nodes from " << graph->global_n() << " to " << result.graph.global_n();

  // Only keep graph if coarsening has not converged yet
  if (!has_converged(*graph, result.graph)) {
    DBG << "... success";

    _graph_hierarchy.push_back(std::move(result.graph));
    _global_mapping_hierarchy.push_back(std::move(result.mapping));
    _node_migration_history.push_back(std::move(result.migration));

    return coarsest();
  }

  DBG << "... converged due to insufficient shrinkage";
  return graph;
}

const DistributedGraph *Coarsener::coarsen_once(const GlobalNodeWeight max_cluster_weight) {
  const DistributedGraph *graph = coarsest();

  if (level() >= _input_ctx.coarsening.max_global_clustering_levels) {
    return graph;
  } else if (level() == _input_ctx.coarsening.max_local_clustering_levels) {
    _local_clustering_converged = true;
  }

  if (!_local_clustering_converged) {
    const DistributedGraph *c_graph = coarsen_once_local(max_cluster_weight);
    if (c_graph == graph) {
      _local_clustering_converged = true;
      // no return -> try global clustering right away
    } else {
      return c_graph;
    }
  }

  return coarsen_once_global(max_cluster_weight);
}

DistributedPartitionedGraph Coarsener::uncoarsen_once(DistributedPartitionedGraph &&p_graph) {
  KASSERT(coarsest() == &p_graph.graph(), "expected graph partition of current coarsest graph");
  KASSERT(!_global_mapping_hierarchy.empty() || !_local_mapping_hierarchy.empty());

  if (!_global_mapping_hierarchy.empty()) {
    return uncoarsen_once_global(std::move(p_graph));
  }

  return uncoarsen_once_local(std::move(p_graph));
}

DistributedPartitionedGraph Coarsener::uncoarsen_once_local(DistributedPartitionedGraph &&p_graph) {
  KASSERT(!_local_mapping_hierarchy.empty(), "", assert::light);

  auto block_weights = p_graph.take_block_weights();
  const DistributedGraph *new_coarsest = nth_coarsest(1);
  const auto &mapping = _local_mapping_hierarchy.back();

  StaticArray<BlockID> partition(new_coarsest->total_n());
  new_coarsest->pfor_all_nodes([&](const NodeID u) { partition[u] = p_graph.block(mapping[u]); });
  const BlockID k = p_graph.k();

  _local_mapping_hierarchy.pop_back();
  _graph_hierarchy.pop_back();

  return {coarsest(), k, std::move(partition), std::move(block_weights)};
}

DistributedPartitionedGraph Coarsener::uncoarsen_once_global(DistributedPartitionedGraph &&p_graph
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

bool Coarsener::has_converged(const DistributedGraph &before, const DistributedGraph &after) const {
  return 1.0 * after.global_n() / before.global_n() >= 0.95;
}

const DistributedGraph *Coarsener::coarsest() const {
  return nth_coarsest(0);
}

std::size_t Coarsener::level() const {
  return _graph_hierarchy.size();
}

const DistributedGraph *Coarsener::nth_coarsest(const std::size_t n) const {
  return _graph_hierarchy.size() > n ? &_graph_hierarchy[_graph_hierarchy.size() - n - 1]
                                     : &_input_graph;
}

GlobalNodeWeight Coarsener::max_cluster_weight() const {
  const auto *graph = coarsest();

  return shm::compute_max_cluster_weight(
      _input_ctx.coarsening,
      graph->global_n(),
      graph->global_total_node_weight(),
      _input_ctx.partition
  );
}
} // namespace kaminpar::dist
