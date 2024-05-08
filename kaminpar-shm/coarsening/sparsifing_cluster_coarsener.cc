/*******************************************************************************
 * cluster corsener with included sparsification
 *
 * @file:   cluster_coarsener.cc
 * @author: Dominik Rosch
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/sparsifing_cluster_coarsener.h"

#include "contraction/cluster_contraction_preprocessing.h"
#include "sparsification/ForestFireSampler.h"
#include "sparsification/UniformRandomSampler.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
SparsifingClusteringCoarsener::SparsifingClusteringCoarsener(
    const Context &ctx, const PartitionContext &p_ctx
)
    : _clustering_algorithm(factory::create_clusterer(ctx)),
      _sampling_algorithm(std::make_unique<sparsification::ForestFireSampler>(
          sparsification::ForestFireSampler(0.3, 0.9, 0.01)
      )),
      _c_ctx(ctx.coarsening),
      _p_ctx(p_ctx) {}

void SparsifingClusteringCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

/**
 * Deletes all edges with weight 0 in the sample and reweights the rest
 * Only the sample entries for an edge (v, u) with v < u are considered
 * @param csr Graph in csr format
 * @param sample for every edge 0, if it should be removed, its (new) weight otherwise
 * @param edges_kept how many edges are samples, i.e., how many entries in sample are not 0
 */
CSRGraph
SparsifingClusteringCoarsener::sparsify(const CSRGraph *g, StaticArray<EdgeWeight> sample) {
  auto nodes = StaticArray<EdgeID>(g->n() + 1);
  for (NodeID v : g->nodes()) {
    for (EdgeID e : g->incident_edges(v)) {
      NodeID u = g->edge_target(e);
      if (v < u && sample[e]) {
        nodes[v + 1]++;
        nodes[u + 1]++;
      }
    }
  }
  parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());

  auto edges_added = StaticArray<EdgeID>(g->n(), 0);
  auto edges = StaticArray<NodeID>(nodes[g->n()]);
  auto edge_weights = StaticArray<EdgeWeight>(nodes[g->n()]);

  for (NodeID v : g->nodes()) {
    for (EdgeID e : g->incident_edges(v)) {
      NodeID u = g->edge_target(e);
      if (v < u && sample[e]) {
        edges[nodes[v] + edges_added[v]] = u;
        edges[nodes[u] + edges_added[u]] = v;
        edge_weights[nodes[v] + edges_added[v]] = sample[e];
        edge_weights[nodes[u] + edges_added[u]] = sample[e];
        edges_added[v]++;
        edges_added[u]++;
      }
    }
  }

  return CSRGraph(
      std::move(nodes),
      std::move(edges),
      std::move(StaticArray<NodeWeight>(g->raw_node_weights().begin(), g->raw_node_weights().end())
      ),
      std::move(edge_weights)
  );
}

bool SparsifingClusteringCoarsener::coarsen() {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  START_HEAP_PROFILER("Allocation");
  RECORD("clustering") StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeWeight total_node_weight = current().total_node_weight();
  const NodeID prev_n = current().n();

  START_HEAP_PROFILER("Label Propagation");
  START_TIMER("Label Propagation");
  _clustering_algorithm->set_max_cluster_weight(
      compute_max_cluster_weight<NodeWeight>(_c_ctx, _p_ctx, prev_n, total_node_weight)
  );
  _clustering_algorithm->set_desired_cluster_count(0);
  _clustering_algorithm->compute_clustering(clustering, current(), free_allocated_memory);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Contract graph");
  auto coarsened = TIMED_SCOPE("Contract graph") {
    return contract_clustering(
        current(), std::move(clustering), _c_ctx.contraction, _contraction_m_ctx
    );
  };

  KASSERT(coarsened->get().m() % 2 == 0, "graph should be undirected", assert::always);
  const CSRGraph *csr = dynamic_cast<const CSRGraph *>(coarsened->get().underlying_graph());
  KASSERT(csr != nullptr, "can only be used with a CSRGraph", assert::always);

  auto sample = _sampling_algorithm->sample(*csr);
  CSRGraph sparsified = sparsify(csr, std::move(sample));

  _hierarchy.push_back(std::make_unique<contraction::CoarseGraphImpl<StaticArray>>(
      Graph(std::make_unique<CSRGraph>(std::move(sparsified))),
      std::move(
          dynamic_cast<contraction::CoarseGraphImpl<StaticArray> *>(coarsened.get())->get_mapping()
      )
  ));
  STOP_HEAP_PROFILER();

  const NodeID next_n = current().n();
  const bool converged = (1.0 - 1.0 * next_n / prev_n) <= _c_ctx.convergence_threshold;

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return !converged;
}

PartitionedGraph SparsifingClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  const BlockID p_graph_k = p_graph.k();
  const auto p_graph_partition = p_graph.take_raw_partition();

  auto coarsened = pop_hierarchy(std::move(p_graph));
  const NodeID next_n = current().n();

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  RECORD("partition") StaticArray<BlockID> partition(next_n);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Project partition");
  coarsened->project(p_graph_partition, partition);
  STOP_TIMER();

  SCOPED_HEAP_PROFILER("Create graph");
  SCOPED_TIMER("Create graph");
  return {current(), p_graph_k, std::move(partition)};
}

std::unique_ptr<CoarseGraph> SparsifingClusteringCoarsener::pop_hierarchy(PartitionedGraph &&p_graph
) {
  KASSERT(!empty(), "cannot pop from an empty graph hierarchy", assert::light);

  auto coarsened = std::move(_hierarchy.back());
  _hierarchy.pop_back();

  KASSERT(
      &coarsened->get() == &p_graph.graph(),
      "p_graph wraps a different graph (ptr="
          << &p_graph.graph() << ") than the one that was coarsened (ptr=" << &coarsened->get()
          << ")",
      assert::light
  );

  return coarsened;
}

bool SparsifingClusteringCoarsener::keep_allocated_memory() const {
  return level() >= _c_ctx.clustering.max_mem_free_coarsening_level;
}
} // namespace kaminpar::shm
