/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar {

using namespace shm;

namespace {

void print_statistics(
    const Context &ctx,
    const PartitionedGraph &p_graph,
    const int max_timer_depth,
    const bool parseable
) {
  const EdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, ctx.partition);

  cio::print_delimiter("Result Summary");

  // statistics output that is easy to parse
  if (parseable) {
    LOG << "RESULT cut=" << cut << " imbalance=" << imbalance << " feasible=" << feasible
        << " k=" << p_graph.k();
#ifdef KAMINPAR_ENABLE_TIMERS
    LLOG << "TIME ";
    Timer::global().print_machine_readable(std::cout);
#else  // KAMINPAR_ENABLE_TIMERS
    LOG << "TIME disabled";
#endif // KAMINPAR_ENABLE_TIMERS
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
    LOG << "MEMORY " << heap_profiler::HeapProfiler::global().peak_memory();
#else
    LOG << "MEMORY disabled";
#endif
    LOG;
  }

#ifdef KAMINPAR_ENABLE_TIMERS
  Timer::global().print_human_readable(std::cout, max_timer_depth);
#else  // KAMINPAR_ENABLE_TIMERS
  LOG << "Global Timers: disabled";
#endif // KAMINPAR_ENABLE_TIMERS
  LOG;
  PRINT_HEAP_PROFILE(std::cout);
  LOG << "Partition summary:";
  if (p_graph.k() != ctx.partition.k) {
    LOG << logger::RED << "  Number of blocks: " << p_graph.k();
  } else {
    LOG << "  Number of blocks: " << p_graph.k();
  }
  LOG << "  Edge cut:         " << cut;
  LOG << "  Imbalance:        " << imbalance;
  if (feasible) {
    LOG << "  Feasible:         yes";
  } else {
    LOG << logger::RED << "  Feasible:         no";
  }
}

} // namespace

KaMinPar::KaMinPar(const int num_threads, Context ctx)
    : _num_threads(num_threads),
      _ctx(std::move(ctx)),
      _gc(tbb::global_control::max_allowed_parallelism, num_threads) {
#ifdef KAMINPAR_ENABLE_TIMERS
  GLOBAL_TIMER.reset();
#endif // KAMINPAR_ENABLE_TIMERS
}

KaMinPar::~KaMinPar() = default;

void KaMinPar::set_output_level(const OutputLevel output_level) {
  _output_level = output_level;
}

void KaMinPar::set_max_timer_depth(const int max_timer_depth) {
  _max_timer_depth = max_timer_depth;
}

Context &KaMinPar::context() {
  return _ctx;
}

void KaMinPar::borrow_and_mutate_graph(
    const NodeID n, EdgeID *xadj, NodeID *adjncy, NodeWeight *vwgt, EdgeWeight *adjwgt
) {
  SCOPED_HEAP_PROFILER("Borrow and mutate graph");
  SCOPED_TIMER("IO");

  const EdgeID m = xadj[n];

  RECORD("nodes") StaticArray<EdgeID> nodes(n + 1, xadj);
  RECORD("edges") StaticArray<NodeID> edges(m, adjncy);
  RECORD("node_weights")
  StaticArray<NodeWeight> node_weights =
      (vwgt == nullptr) ? StaticArray<NodeWeight>(0) : StaticArray<NodeWeight>(n, vwgt);
  RECORD("edge_weights")
  StaticArray<EdgeWeight> edge_weights =
      (adjwgt == nullptr) ? StaticArray<EdgeWeight>(0) : StaticArray<EdgeWeight>(m, adjwgt);

  auto csr_graph = std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), false
  );
  KASSERT(shm::debug::validate_graph(*csr_graph), "invalid input graph", assert::heavy);
  set_graph(Graph(std::move(csr_graph)));
}

void KaMinPar::copy_graph(
    const NodeID n,
    const EdgeID *const xadj,
    const NodeID *const adjncy,
    const NodeWeight *const vwgt,
    const EdgeWeight *const adjwgt
) {
  SCOPED_HEAP_PROFILER("Copy graph");
  SCOPED_TIMER("IO");

  const EdgeID m = xadj[n];
  const bool has_node_weights = vwgt != nullptr;
  const bool has_edge_weights = adjwgt != nullptr;

  RECORD("nodes") StaticArray<EdgeID> nodes(n + 1);
  RECORD("edges") StaticArray<NodeID> edges(m);
  RECORD("node_weights") StaticArray<NodeWeight> node_weights(has_node_weights ? n : 0);
  RECORD("edge_weights") StaticArray<EdgeWeight> edge_weights(has_edge_weights ? m : 0);

  nodes[n] = xadj[n];
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    nodes[u] = xadj[u];
    if (has_node_weights) {
      node_weights[u] = vwgt[u];
    }
  });
  tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
    edges[e] = adjncy[e];
    if (has_edge_weights) {
      edge_weights[e] = adjwgt[e];
    }
  });

  auto csr_graph = std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), false
  );
  KASSERT(shm::debug::validate_graph(*csr_graph), "invalid input graph", assert::heavy);
  set_graph(Graph(std::move(csr_graph)));
}

void KaMinPar::set_graph(Graph graph) {
  _was_rearranged = false;
  _graph_ptr = std::make_unique<Graph>(std::move(graph));
}

void KaMinPar::reseed(int seed) {
  Random::reseed(seed);
}

EdgeWeight KaMinPar::compute_partition(
    const BlockID k, BlockID *partition, const bool use_initial_node_ordering
) {
  if (_output_level == OutputLevel::QUIET) {
    Logger::set_quiet_mode(true);
  }

  cio::print_kaminpar_banner();
  cio::print_build_identifier();
  cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();
  cio::print_delimiter("Input Summary", '#');

  const double original_epsilon = _ctx.partition.epsilon;
  _ctx.parallel.num_threads = _num_threads;
  _ctx.partition.k = k;

  // Setup graph dependent context parameters
  _ctx.setup(*_graph_ptr);

  // Initialize console output
  if (_output_level >= OutputLevel::APPLICATION) {
    print(_ctx, std::cout);
  }

  START_HEAP_PROFILER("Partitioning");
  START_TIMER("Partitioning");

  if (!_was_rearranged) {
    if (_ctx.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
      if (_ctx.compression.enabled) {
        LOG_WARNING << "A compressed graph cannot be rearranged by degree buckets. Disabling "
                       "degree bucket ordering!";
        _ctx.node_ordering = NodeOrdering::NATURAL;
      } else if (!_graph_ptr->sorted()) {
        CSRGraph &csr_graph = _graph_ptr->csr_graph();
        _graph_ptr = std::make_unique<Graph>(graph::rearrange_by_degree_buckets(csr_graph));
      }
    }

    if (_ctx.edge_ordering == EdgeOrdering::COMPRESSION && !_ctx.compression.enabled) {
      CSRGraph &csr_graph = _graph_ptr->csr_graph();
      graph::reorder_edges_by_compression(csr_graph);
    }

    _was_rearranged = true;
  }

  // Cut off isolated nodes if the graph has been rearranged such that the isolated nodes are placed
  // at the end.
  if (_graph_ptr->sorted()) {
    graph::remove_isolated_nodes(*_graph_ptr, _ctx.partition);
  }

  // Perform actual partitioning
  PartitionedGraph p_graph = [&] {
    auto partitioner = factory::create_partitioner(*_graph_ptr, _ctx);
    if (_output_level >= OutputLevel::DEBUG) {
      partitioner->enable_metrics_output();
    }
    PartitionedGraph p_graph = partitioner->partition();

    START_TIMER("Deallocation");
    partitioner.reset();
    STOP_TIMER();

    return p_graph;
  }();

  // Re-integrate isolated nodes that were cut off during preprocessing
  if (_graph_ptr->sorted()) {
    SCOPED_HEAP_PROFILER("Re-integrate isolated nodes");
    SCOPED_TIMER("Re-integrate isolated nodes");

    const NodeID num_isolated_nodes =
        graph::integrate_isolated_nodes(*_graph_ptr, original_epsilon, _ctx);
    p_graph = graph::assign_isolated_nodes(std::move(p_graph), num_isolated_nodes, _ctx.partition);
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("IO");
  if (_graph_ptr->permuted() && use_initial_node_ordering) {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(_graph_ptr->map_original_node(u));
    });
  } else {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(u);
    });
  }
  STOP_TIMER();

  // Print some statistics
  STOP_TIMER(); // stop root timer
  if (_output_level >= OutputLevel::APPLICATION) {
    print_statistics(_ctx, p_graph, _max_timer_depth, _output_level >= OutputLevel::EXPERIMENT);
  }

  const EdgeWeight final_cut = metrics::edge_cut(p_graph);

#ifdef KAMINPAR_ENABLE_TIMERS
  GLOBAL_TIMER.reset();
#endif // KAMINPAR_ENABLE_TIMERS

  return final_cut;
}

const shm::Graph *KaMinPar::graph() {
  return _graph_ptr.get();
}

} // namespace kaminpar
