/*******************************************************************************
 * Deep multilevel graph partitioning scheme.
 *
 * @file:   deep_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/partitioning/debug.h"
#include "kaminpar-shm/partitioning/deep/async_initial_partitioning.h"
#include "kaminpar-shm/partitioning/deep/sync_initial_partitioning.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(true);

} // namespace

using namespace partitioning;

DeepMultilevelPartitioner::DeepMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx),
      _current_p_ctx(input_ctx.partition),
      _coarsener(factory::create_coarsener(input_ctx)),
      _refiner(factory::create_refiner(input_ctx)),
      _bipartitioner_pool(_input_ctx) {
  _coarsener->initialize(&_input_graph);
  _refiner->set_output_level(Refiner::OutputLevel::INFO);
  _refiner->set_output_prefix("   ");
}

void DeepMultilevelPartitioner::use_communities(
    const std::span<const NodeID> communities, const NodeID num_communities
) {
  _coarsener->use_communities(communities);
  _num_communities = num_communities;
}

PartitionedGraph DeepMultilevelPartitioner::partition() {
  cio::print_delimiter("Partitioning");

  if (_print_metrics) {
    _refiner->set_output_level(Refiner::OutputLevel::DEBUG);
    _refiner->set_output_prefix("    ");
  }

  return uncoarsen(initial_partition(coarsen()));
}

const Graph *DeepMultilevelPartitioner::coarsen() {
  SCOPED_HEAP_PROFILER("Coarsening");
  SCOPED_TIMER("Coarsening");

  const Graph *c_graph = &_input_graph;
  NodeID prev_c_graph_n = c_graph->n();
  EdgeID prev_c_graph_m = c_graph->m();
  NodeWeight prev_c_graph_total_node_weight = c_graph->total_node_weight();

  bool shrunk = true;
  bool search_subgraph_memory_size = true;

  while (shrunk && c_graph->n() > initial_partitioning_threshold()) {
    // If requested, dump graph before each coarsening step + after coarsening
    // converged. This way, we also have a dump of the (reordered) input graph,
    // which makes it easier to use the final partition (before reordering it).
    // We dump the coarsest graph in ::initial_partitioning().
    debug::dump_graph_hierarchy(*c_graph, _coarsener->level(), _input_ctx);

    // Store the size of the previous coarse graph, so that we can pre-allocate _subgraph_memory
    // if we need it for this graph (see below)
    prev_c_graph_n = c_graph->n();
    prev_c_graph_m = c_graph->m();
    prev_c_graph_total_node_weight = c_graph->total_node_weight();

    // Build next coarse graph
    shrunk = _coarsener->coarsen();
    c_graph = &_coarsener->current();

    // _subgraph_memory stores the block-induced subgraphs of the partitioned graph during recursive
    // bipartitioning
    // To avoid repeated allocation, we pre-allocate the memory during coarsening for the largest
    // coarse graph for which we still need recursive bipartitioning
    if (search_subgraph_memory_size &&
        partitioning::compute_k_for_n(c_graph->n(), _input_ctx) < _input_ctx.partition.k) {
      search_subgraph_memory_size = false;

      _last_initial_partitioning_level = _coarsener->level() - 1;

      _subgraph_memory_n = prev_c_graph_n;
      _subgraph_memory_m = prev_c_graph_m;

      const bool toplevel = _coarsener->level() == 1;
      if (toplevel && !_input_graph.is_node_weighted()) {
        _subgraph_memory_n_weights = c_graph->n();
      } else {
        _subgraph_memory_n_weights = prev_c_graph_n;
      }

      if (toplevel && !_input_graph.is_edge_weighted()) {
        _subgraph_memory_m_weights = c_graph->m();
      } else {
        _subgraph_memory_m_weights = prev_c_graph_m;
      }
    }

    // Print some metrics for the coarse graphs
    LOG << "Coarsening -> Level " << _coarsener->level();
    LOG << " Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
    LOG << " Maximum node weight: " << c_graph->max_node_weight() << " <= "
        << compute_max_cluster_weight<NodeWeight>(
               _input_ctx.coarsening,
               _input_ctx.partition,
               prev_c_graph_n,
               prev_c_graph_total_node_weight
           );
    LOG;
  }

  if (search_subgraph_memory_size) {
    _subgraph_memory_n = _subgraph_memory_n_weights = prev_c_graph_n;
    _subgraph_memory_m = _subgraph_memory_m_weights = prev_c_graph_m;
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

NodeID DeepMultilevelPartitioner::initial_partitioning_threshold() {
  const auto mode = _input_ctx.partitioning.deep_initial_partitioning_mode;
  const bool is_parallel_mode =
      (mode == InitialPartitioningMode::SYNCHRONOUS_PARALLEL ||
       mode == InitialPartitioningMode::ASYNCHRONOUS_PARALLEL);

  if (is_parallel_mode) { // Parallel: copy for each thread once n <= p * C
    return _input_ctx.parallel.num_threads * _input_ctx.coarsening.contraction_limit;
  } else if (mode == InitialPartitioningMode::COMMUNITIES) {
    return _input_ctx.coarsening.contraction_limit * _num_communities;
  } else { // Sequential: coarsen until until n <= 2 * C
    return 2 * _input_ctx.coarsening.contraction_limit;
  }
}

PartitionedGraph DeepMultilevelPartitioner::initial_partition(const Graph *graph) {
  SCOPED_HEAP_PROFILER("Initial partitioning");
  SCOPED_TIMER("Initial partitioning scheme");
  LOG << "Initial partitioning:";

  if (!_input_ctx.partitioning.use_lazy_subgraph_memory) {
    SCOPED_HEAP_PROFILER("SubgraphMemory resize");
    SCOPED_TIMER("Allocation");

    _subgraph_memory.weighted_resize(
        _subgraph_memory_n,
        _input_ctx.partition.k,
        _subgraph_memory_m,
        _subgraph_memory_n_weights,
        _subgraph_memory_m_weights
    );
  }

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
  PartitionedGraph p_graph = [&] {
    switch (_input_ctx.partitioning.deep_initial_partitioning_mode) {
    case InitialPartitioningMode::SEQUENTIAL:
      return _bipartitioner_pool.bipartition(graph, 0, 1, true);

    case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
      return SyncInitialPartitioner(_input_ctx, _bipartitioner_pool, _tmp_extraction_mem_pool_ets)
          .partition(_coarsener.get(), _current_p_ctx);

    case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
      return AsyncInitialPartitioner(_input_ctx, _bipartitioner_pool, _tmp_extraction_mem_pool_ets)
          .partition(_coarsener.get(), _current_p_ctx);

    case InitialPartitioningMode::COMMUNITIES:
      return initial_partition_by_communities(graph);
    }

    __builtin_unreachable();
  }();
  ENABLE_TIMERS();

  _current_p_ctx = create_kway_context(_input_ctx, p_graph);
  DBG << debug::describe_partition_state(p_graph, _current_p_ctx);

  // Print some metrics for the initial partition.
  LOG << " Number of blocks: " << p_graph.k();
  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << " Cut:              " << metrics::edge_cut(p_graph);
    LOG << " Imbalance:        " << metrics::imbalance(p_graph);
    LOG << " Feasible:         " << (metrics::is_feasible(p_graph, _current_p_ctx) ? "yes" : "no");
  }

  // If requested, dump the coarsest partition -- as noted above, this is not
  // actually the coarsest partition when using deep multilevel.
  debug::dump_coarsest_partition(p_graph, _input_ctx);
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "post-refinement", _input_ctx);

  return p_graph;
}

StaticArray<BlockID> DeepMultilevelPartitioner::copy_coarsest_communities() {
  std::span<const NodeID> communities = _coarsener->current_communities();
  return {communities.begin(), communities.end()};
}

PartitionedGraph DeepMultilevelPartitioner::initial_partition_by_communities(const Graph *graph) {
  StaticArray<BlockID> partition = copy_coarsest_communities();
  KASSERT(partition.size() == graph->n());

  PartitionedGraph p_graph(*graph, static_cast<BlockID>(_num_communities), std::move(partition));
  return p_graph;
}

PartitionedGraph DeepMultilevelPartitioner::uncoarsen(PartitionedGraph p_graph) {
  SCOPED_HEAP_PROFILER("Uncoarsening");

  bool refined = false;
  while (!_coarsener->empty()) {
    SCOPED_HEAP_PROFILER("Level", std::to_string(_coarsener->level() - 1));

    LOG;
    LOG << "Uncoarsening -> Level " << (_coarsener->level() - 1);

    p_graph = _coarsener->uncoarsen(std::move(p_graph));
    _current_p_ctx = create_kway_context(_input_ctx, p_graph);

    LOG << " Number of nodes: " << p_graph.n() << " | Number of edges: " << p_graph.m();

    refine(p_graph);
    refined = true;

    const BlockID desired_k = partitioning::compute_k_for_n(p_graph.n(), _input_ctx);
    if (p_graph.k() < desired_k) {
      extend_partition(p_graph, desired_k);
      refined = false;

      if (_input_ctx.partitioning.refine_after_extending_partition) {
        refine(p_graph);
        refined = true;
      }
    }
  }

  _current_p_ctx = create_kway_context(_input_ctx, p_graph);

  if (!refined || p_graph.k() < _input_ctx.partition.k) {
    SCOPED_HEAP_PROFILER("Toplevel");

    LOG;
    LOG << "Toplevel:";
    LOG << " Number of nodes: " << p_graph.n() << " | Number of edges: " << p_graph.m();

    if (!refined) {
      refine(p_graph);
    }
    if (p_graph.k() < _input_ctx.partition.k) {
      extend_partition(p_graph, _input_ctx.partition.k);
      refine(p_graph);
    }
  }

  return p_graph;
}

void DeepMultilevelPartitioner::refine(PartitionedGraph &p_graph) {
  SCOPED_HEAP_PROFILER("Refinement");
  SCOPED_TIMER("Refinement");

  if (_input_ctx.partitioning.restrict_vcycle_refinement && _num_communities > 0) {
    _refiner->set_communities(_coarsener->current_communities());
  }

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

void DeepMultilevelPartitioner::extend_partition(PartitionedGraph &p_graph, const BlockID k_prime) {
  SCOPED_HEAP_PROFILER("Extending partition");
  LOG << "  Extending partition from " << p_graph.k() << " blocks to " << k_prime << " blocks";

  DBG << "Partition state before extending partition from " << p_graph.k() << " to " << k_prime
      << " blocks:";
  DBG << debug::describe_partition_state(p_graph, _current_p_ctx);

  // If we are in a v-cycle, we might have a partition whose number of blocks is not a power of two
  // -- something on which the other extend_* functions rely. In this case, first "complete" the
  // current level of the recursion tree to obtain a power-of-two block count.
  if (_input_ctx.partitioning.deep_initial_partitioning_mode ==
      InitialPartitioningMode::COMMUNITIES) {
    partitioning::complete_partial_extend_partition(
        p_graph,
        _input_ctx,
        _extraction_mem_pool_ets,
        _tmp_extraction_mem_pool_ets,
        _bipartitioner_pool
    );
    _current_p_ctx = create_kway_context(_input_ctx, p_graph);
  }

  if (_input_ctx.partitioning.use_lazy_subgraph_memory) {
    partitioning::extend_partition_lazy_extraction(
        p_graph,
        k_prime,
        _input_ctx,
        _extraction_mem_pool_ets,
        _tmp_extraction_mem_pool_ets,
        _bipartitioner_pool,
        _input_ctx.parallel.num_threads
    );
  } else {
    partitioning::extend_partition(
        p_graph,
        k_prime,
        _input_ctx,
        _subgraph_memory,
        _tmp_extraction_mem_pool_ets,
        _bipartitioner_pool,
        _input_ctx.parallel.num_threads
    );
  }

  if (_last_initial_partitioning_level == _coarsener->level()) {
    SCOPED_TIMER("Deallocation");
    _subgraph_memory.free();
    _extraction_mem_pool_ets.clear();
    _tmp_extraction_mem_pool_ets.clear();
    _bipartitioner_pool.free();
  }

  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "   Cut:       " << metrics::edge_cut(p_graph);
    LOG << "   Imbalance: " << metrics::imbalance(p_graph);
  }

  _current_p_ctx = create_kway_context(_input_ctx, p_graph);
  DBG << "Partition state after extending partition to " << k_prime << " blocks:";
  DBG << debug::describe_partition_state(p_graph, _current_p_ctx);
}

} // namespace kaminpar::shm
