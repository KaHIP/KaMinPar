/*******************************************************************************
 * k-way multilevel graph partitioning scheme.
 *
 * @file:   kway_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   19.09.2023
 ******************************************************************************/
#include "kaminpar-shm/partitioning/kway/kway_multilevel.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/partitioning/debug.h"
#include "kaminpar-shm/partitioning/helper.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();

} // namespace

using namespace partitioning;

KWayMultilevelPartitioner::KWayMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx),
      _current_p_ctx(input_ctx.partition),
      _coarsener(factory::create_coarsener(input_ctx)),
      _refiner(factory::create_refiner(input_ctx)),
      _bipartitioner_pool(_input_ctx) {
  _coarsener->initialize(&_input_graph);
}

PartitionedGraph KWayMultilevelPartitioner::partition() {
  cio::print_delimiter("Partitioning");
  return uncoarsen(initial_partition(coarsen()));
}

void KWayMultilevelPartitioner::refine(PartitionedGraph &p_graph) {
  SCOPED_HEAP_PROFILER("Refinement");
  SCOPED_TIMER("Refinement");

  // If requested, dump the current partition to disk before refinement ...
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "pre-refinement", _input_ctx);

  LOG << "  Running refinement on " << p_graph.k() << " blocks";
  _refiner->initialize(p_graph);
  _refiner->refine(p_graph, _current_p_ctx);

  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "   Cut:       " << metrics::edge_cut(p_graph);
    LOG << "   Imbalance: " << metrics::imbalance(p_graph);
    LOG << "   Feasible:  " << metrics::is_feasible(p_graph, _current_p_ctx);
  }

  // ... and dump it after refinement.
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "post-refinement", _input_ctx);
}

PartitionedGraph KWayMultilevelPartitioner::uncoarsen(PartitionedGraph p_graph) {
  SCOPED_HEAP_PROFILER("Uncoarsening");
  SCOPED_TIMER("Uncoarsening");

  refine(p_graph);

  while (!_coarsener->empty()) {
    SCOPED_TIMER("Level", std::to_string(_coarsener->level() - 1));

    LOG;
    LOG << "Uncoarsening -> Level " << _coarsener->level() - 1;

    p_graph = _coarsener->uncoarsen(std::move(p_graph));
    _current_p_ctx = create_kway_context(_input_ctx, p_graph);

    refine(p_graph);
  }

  return p_graph;
}

const Graph *KWayMultilevelPartitioner::coarsen() {
  SCOPED_HEAP_PROFILER("Coarsening");
  SCOPED_TIMER("Coarsening");

  const Graph *c_graph = &_input_graph;
  bool shrunk = true;

  LOG << "Input graph:";
  LOG << " Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
  LOG << " Maximum node weight: " << c_graph->max_node_weight();
  LOG;

  while (shrunk && c_graph->n() > initial_partitioning_threshold()) {
    SCOPED_TIMER("Level", std::to_string(_coarsener->level()));

    // If requested, dump graph before each coarsening step + after coarsening
    // converged. This way, we also have a dump of the (reordered) input graph,
    // which makes it easier to use the final partition (before reordering it).
    // We dump the coarsest graph in ::initial_partitioning().
    debug::dump_graph_hierarchy(*c_graph, _coarsener->level(), _input_ctx);

    const NodeID prev_c_graph_n = c_graph->n();
    const NodeWeight prev_c_graph_total_node_weight = c_graph->total_node_weight();

    // Build next coarse graph
    shrunk = _coarsener->coarsen();
    c_graph = &_coarsener->current();

    // Print some metrics for the coarse graphs
    const NodeWeight max_cluster_weight = compute_max_cluster_weight<NodeWeight>(
        _input_ctx.coarsening, _input_ctx.partition, prev_c_graph_n, prev_c_graph_total_node_weight
    );
    LOG << "Coarsening -> Level " << _coarsener->level()
        << " [max cluster weight: " << max_cluster_weight << "]:";
    LOG << "  Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
    LOG << "  Maximum node weight: " << c_graph->max_node_weight() << " ";
    LOG;
  }

  _coarsener->release_allocated_memory();

  if (shrunk) {
    LOG << "==> Coarsening terminated with less than " << initial_partitioning_threshold()
        << " nodes";
    LOG;
  } else {
    LOG << "==> Coarsening converged";
    LOG;
  }

  return c_graph;
}

NodeID KWayMultilevelPartitioner::initial_partitioning_threshold() {
  return _input_ctx.partition.k * _input_ctx.coarsening.contraction_limit;
}

PartitionedGraph KWayMultilevelPartitioner::initial_partition(const Graph *graph) {
  SCOPED_HEAP_PROFILER("Initial partitioning");
  SCOPED_TIMER("Initial partitioning");
  LOG << "Initial partitioning:";

  // If requested, dump the coarsest graph to disk. Note that in the context of
  // deep multilevel, this is not actually the coarsest graph, but rather the
  // coarsest graph before splitting PEs and duplicating the graph.
  // Disable worker splitting with --p-deep-initial-partitioning-mode=sequential to obtain coarser
  // graphs.
  debug::dump_coarsest_graph(*graph, _input_ctx);
  debug::dump_graph_hierarchy(*graph, _coarsener->level(), _input_ctx);

  // Since timers are not multi-threaded, we disable them during (parallel)
  // initial partitioning.
  DISABLE_TIMERS();
  PartitionedGraph p_graph = _bipartitioner_pool.bipartition(graph, 0, 1, true);

  graph::SubgraphMemory subgraph_memory(p_graph.n(), _input_ctx.partition.k, p_graph.m());
  partitioning::TemporarySubgraphMemoryEts ip_extraction_pool_ets;

  partitioning::extend_partition(
      p_graph,
      _input_ctx.partition.k,
      _input_ctx,
      subgraph_memory,
      ip_extraction_pool_ets,
      _bipartitioner_pool,
      _input_ctx.parallel.num_threads
  );

  _current_p_ctx = create_kway_context(_input_ctx, p_graph);

  ENABLE_TIMERS();

  // Print some metrics for the initial partition.
  LOG << "  Number of blocks: " << p_graph.k();
  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "  Cut:              " << metrics::edge_cut(p_graph);
    LOG << "  Imbalance:        " << metrics::imbalance(p_graph);
    LOG << "  Feasible:         " << (metrics::is_feasible(p_graph, _current_p_ctx) ? "yes" : "no");
  }

  // If requested, dump the coarsest partition -- as noted above, this is not
  // actually the coarsest partition when using deep multilevel.
  debug::dump_coarsest_partition(p_graph, _input_ctx);
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "post-refinement", _input_ctx);

  return p_graph;
}

} // namespace kaminpar::shm
