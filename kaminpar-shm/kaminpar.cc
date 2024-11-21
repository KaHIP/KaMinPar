/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/kaminpar.h"

#include <cmath>

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

namespace shm {

double PartitionContext::epsilon() const {
  KASSERT(_max_total_block_weight != kInvalidBlockWeight);
  return (1.0 * _max_total_block_weight / total_node_weight) - 1.0;
}

double PartitionContext::inferred_epsilon() const {
  return epsilon();
}

void PartitionContext::setup(
    const AbstractGraph &graph,
    const BlockID k,
    const double epsilon,
    const bool relax_max_block_weights
) {
  // this->total_node_weight not yet initialized: use graph.total_node_weight instead
  const BlockWeight perfectly_balanced_block_weight =
      std::ceil(1.0 * graph.total_node_weight() / k);
  std::vector<BlockWeight> max_block_weights(k, (1.0 + epsilon) * perfectly_balanced_block_weight);
  setup(graph, std::move(max_block_weights), relax_max_block_weights);
}

void PartitionContext::setup(
    const AbstractGraph &graph,
    std::vector<BlockWeight> max_block_weights,
    const bool relax_max_block_weights
) {
  n = graph.n();
  m = graph.m();
  total_node_weight = graph.total_node_weight();
  total_edge_weight = graph.total_edge_weight();
  max_node_weight = graph.max_node_weight();

  k = static_cast<BlockID>(max_block_weights.size());
  _max_block_weights = std::move(max_block_weights);

  if (relax_max_block_weights) {
    const double eps = epsilon();
    for (BlockWeight &max_block_weight : _max_block_weights) {
      max_block_weight =
          std::max<BlockWeight>(max_block_weight, max_block_weight / (1.0 + eps) + max_node_weight);
    }
  }
}

void GraphCompressionContext::setup(const Graph &graph) {
  high_degree_encoding = CompressedGraph::kHighDegreeEncoding;
  high_degree_threshold = CompressedGraph::kHighDegreeThreshold;
  high_degree_part_length = CompressedGraph::kHighDegreePartLength;
  interval_encoding = CompressedGraph::kIntervalEncoding;
  interval_length_treshold = CompressedGraph::kIntervalLengthTreshold;
  streamvbyte_encoding = CompressedGraph::kStreamVByteEncoding;

  if (enabled) {
    const auto &compressed_graph = graph.compressed_graph();
    compression_ratio = compressed_graph.compression_ratio();
    size_reduction = compressed_graph.size_reduction();
    num_high_degree_nodes = compressed_graph.num_high_degree_nodes();
    num_high_degree_parts = compressed_graph.num_high_degree_parts();
    num_interval_nodes = compressed_graph.num_interval_nodes();
    num_intervals = compressed_graph.num_intervals();
  }
}

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

  // Statistics output that is easy to parse
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

} // namespace shm

using namespace shm;

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

EdgeWeight KaMinPar::compute_partition(const BlockID k, BlockID *partition) {
  return compute_partition(k, 0.03, partition);
}

EdgeWeight KaMinPar::compute_partition(const BlockID k, const double epsilon, BlockID *partition) {
  _ctx.partition.setup(*_graph_ptr, k, epsilon);
  return compute_partition(partition);
}

EdgeWeight
KaMinPar::compute_partition(std::vector<BlockWeight> max_block_weights, BlockID *partition) {
  _ctx.partition.setup(*_graph_ptr, std::move(max_block_weights));
  return compute_partition(partition);
}

EdgeWeight KaMinPar::compute_partition(BlockID *partition) {
  if (_output_level == OutputLevel::QUIET) {
    Logger::set_quiet_mode(true);
  }

  cio::print_kaminpar_banner();
  cio::print_build_identifier();
  cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();
  cio::print_delimiter("Input Summary", '#');

  _ctx.compression.setup(*_graph_ptr);
  _ctx.parallel.num_threads = _num_threads;

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

    const NodeID num_isolated_nodes = _graph_ptr->integrate_isolated_nodes();
    p_graph = graph::assign_isolated_nodes(std::move(p_graph), num_isolated_nodes, _ctx.partition);
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("IO");
  if (_graph_ptr->permuted()) {
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
