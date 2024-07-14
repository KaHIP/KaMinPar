/*******************************************************************************
 * Graph coarsener based on global clusterings.
 *
 * @file:   global_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#include "kaminpar-dist/coarsening/global_cluster_coarsener.h"

#include "kaminpar-dist/coarsening/contraction/global_cluster_contraction.h"
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
      _clusterer(factory::create_clusterer(_input_ctx)) {}

void GlobalClusterCoarsener::initialize(const DistributedGraph *graph) {
  _input_graph = graph;
  _graph_hierarchy.clear();
}

bool GlobalClusterCoarsener::coarsen() {
  DBG << "Coarsen graph using global clustering algorithm ...";

  const DistributedGraph &graph = current();

  RECORD("clustering") StaticArray<GlobalNodeID> clustering(graph.total_n(), static_array::noinit);
  _clusterer->set_max_cluster_weight(max_cluster_weight());
  _clusterer->cluster(clustering, graph);

  auto coarse_graph = contract_clustering(graph, clustering, _input_ctx.coarsening);
  KASSERT(
      debug::validate_graph(coarse_graph->get()),
      "invalid graph after global cluster contraction",
      assert::heavy
  );

  if (!has_converged(graph, coarse_graph->get())) {
    DBG << "... accepted coarsened graph";

    _graph_hierarchy.push_back(std::move(coarse_graph));
    return true;
  }

  DBG << "... converged due to insufficient shrinkage, discarding last coarsening step";
  return false;
}

DistributedPartitionedGraph
GlobalClusterCoarsener::uncoarsen(DistributedPartitionedGraph &&p_c_graph) {
  std::unique_ptr<CoarseGraph> c_graph = std::move(_graph_hierarchy.back());
  KASSERT(
      &c_graph->get() == &p_c_graph.graph(),
      "given graph partition does not belong to the coarse graph"
  );

  _graph_hierarchy.pop_back();
  const DistributedGraph &f_graph = current();

  RECORD("partition") StaticArray<BlockID> f_partition(f_graph.total_n(), static_array::noinit);
  c_graph->project(p_c_graph.partition(), f_partition);

  DistributedPartitionedGraph p_f_graph(
      &f_graph, p_c_graph.k(), std::move(f_partition), p_c_graph.take_block_weights()
  );
  KASSERT(
      debug::validate_partition(p_f_graph),
      "invalid partition after projection to finer graph",
      assert::heavy
  );

  return p_f_graph;
}

bool GlobalClusterCoarsener::has_converged(
    const DistributedGraph &before, const DistributedGraph &after
) const {
  return 1.0 * after.global_n() / before.global_n() >= 0.95;
}

const DistributedGraph &GlobalClusterCoarsener::current() const {
  return _graph_hierarchy.empty() ? *_input_graph : _graph_hierarchy.back()->get();
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
