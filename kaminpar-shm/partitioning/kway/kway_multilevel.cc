/*******************************************************************************
 * k-way multilevel graph partitioning scheme.
 *
 * @file:   kway_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   19.09.2023
 ******************************************************************************/
#include "kaminpar-shm/partitioning/kway/kway_multilevel.h"

#include "kaminpar-shm/partitioning/debug.h"

#include "kaminpar-common/console_io.h"

namespace kaminpar::shm {
using namespace partitioning;

KWayMultilevelPartitioner::KWayMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx),
      _current_p_ctx(input_ctx.partition),
      _coarsener(factory::create_coarsener(input_graph, input_ctx.coarsening)),
      _refiner(factory::create_refiner(input_ctx)) {}

PartitionedGraph KWayMultilevelPartitioner::partition() {
  cio::print_delimiter("Partitioning");
  return uncoarsen(initial_partition(coarsen()));
}

void KWayMultilevelPartitioner::refine(PartitionedGraph &p_graph) {
  // If requested, dump the current partition to disk before refinement ...
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "pre-refinement", _input_ctx);

  helper::refine(_refiner.get(), p_graph, _current_p_ctx);
  LOG << "  Cut:       " << metrics::edge_cut(p_graph);
  LOG << "  Imbalance: " << metrics::imbalance(p_graph);
  LOG << "  Feasible:  " << metrics::is_feasible(p_graph, _current_p_ctx);

  // ... and dump it after refinement.
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "post-refinement", _input_ctx);
}

PartitionedGraph KWayMultilevelPartitioner::uncoarsen(PartitionedGraph p_graph) {
  refine(p_graph);

  while (!_coarsener->empty()) {
    LOG;
    LOG << "Uncoarsening -> Level " << _coarsener.get()->size();
    p_graph = helper::uncoarsen_once(
        _coarsener.get(), std::move(p_graph), _current_p_ctx, _input_ctx.partition
    );
    refine(p_graph);
  }

  return p_graph;
}

const Graph *KWayMultilevelPartitioner::coarsen() {
  const Graph *c_graph = &_input_graph;
  bool shrunk = true;

  while (shrunk && c_graph->n() > initial_partitioning_threshold()) {
    // If requested, dump graph before each coarsening step + after coarsening
    // converged. This way, we also have a dump of the (reordered) input graph,
    // which makes it easier to use the final partition (before reordering it).
    // We dump the coarsest graph in ::initial_partitioning().
    debug::dump_graph_hierarchy(*c_graph, _coarsener->size(), _input_ctx);

    // Build next coarse graph
    shrunk = helper::coarsen_once(_coarsener.get(), c_graph, _input_ctx, _current_p_ctx);
    c_graph = _coarsener->coarsest_graph();

    // Print some metrics for the coarse graphs
    const NodeWeight max_cluster_weight =
        compute_max_cluster_weight(_input_ctx.coarsening, *c_graph, _input_ctx.partition);
    LOG << "Coarsening -> Level " << _coarsener.get()->size();
    LOG << "  Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
    LOG << "  Maximum node weight: " << c_graph->max_node_weight() << " <= " << max_cluster_weight;
    LOG;
  }

  if (shrunk) {
    LOG << "==> Coarsening terminated with less than " << initial_partitioning_threshold()
        << " nodes.";
    LOG;
  } else {
    LOG << "==> Coarsening converged.";
    LOG;
  }

  return c_graph;
}

NodeID KWayMultilevelPartitioner::initial_partitioning_threshold() {
  return _input_ctx.partition.k * _input_ctx.coarsening.contraction_limit;
}

PartitionedGraph KWayMultilevelPartitioner::initial_partition(const Graph *graph) {
  SCOPED_TIMER("Initial partitioning");
  LOG << "Initial partitioning:";

  // If requested, dump the coarsest graph to disk. Note that in the context of
  // deep multilevel, this is not actually the coarsest graph, but rather the
  // coarsest graph before splitting PEs and duplicating the graph.
  // Disable worker splitting with --p-deep-initial-partitioning-mode=sequential to obtain coarser
  // graphs.
  debug::dump_coarsest_graph(*graph, _input_ctx);
  debug::dump_graph_hierarchy(*graph, _coarsener->size(), _input_ctx);

  // Since timers are not multi-threaded, we disable them during (parallel)
  // initial partitioning.
  DISABLE_TIMERS();
  PartitionedGraph p_graph =
      helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);
  helper::update_partition_context(_current_p_ctx, p_graph, _input_ctx.partition.k);

  graph::SubgraphMemory subgraph_memory(p_graph.n(), _input_ctx.partition.k, p_graph.m());
  partitioning::TemporaryGraphExtractionBufferPool ip_extraction_pool;

  helper::extend_partition(
      p_graph,
      _input_ctx.partition.k,
      _input_ctx,
      _current_p_ctx,
      subgraph_memory,
      ip_extraction_pool,
      _ip_m_ctx_pool
  );

  helper::update_partition_context(_current_p_ctx, p_graph, _input_ctx.partition.k);
  ENABLE_TIMERS();

  // Print some metrics for the initial partition.
  LOG << "  Cut:              " << metrics::edge_cut(p_graph);
  LOG << "  Imbalance:        " << metrics::imbalance(p_graph);
  LOG << "  Feasible:         " << (metrics::is_feasible(p_graph, _current_p_ctx) ? "yes" : "no");

  // If requested, dump the coarsest partition -- as noted above, this is not
  // actually the coarsest partition when using deep multilevel.
  debug::dump_coarsest_partition(p_graph, _input_ctx);
  debug::dump_partition_hierarchy(p_graph, _coarsener->size(), "post-refinement", _input_ctx);

  return p_graph;
}
} // namespace kaminpar::shm
