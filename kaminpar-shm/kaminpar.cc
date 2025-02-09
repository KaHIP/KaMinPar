/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/kaminpar.h"

#include <algorithm>
#include <cmath>

#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar {

namespace shm {

void PartitionContext::setup(
    const AbstractGraph &graph,
    const BlockID k,
    const double epsilon,
    const bool relax_max_block_weights
) {
  _epsilon = epsilon;

  // this->total_node_weight not yet initialized: use graph.total_node_weight instead
  const BlockWeight perfectly_balanced_block_weight =
      std::ceil(1.0 * graph.total_node_weight() / k);
  std::vector<BlockWeight> max_block_weights(k, (1.0 + epsilon) * perfectly_balanced_block_weight);
  setup(graph, std::move(max_block_weights), relax_max_block_weights);

  _uniform_block_weights = true;
}

void PartitionContext::setup(
    const AbstractGraph &graph,
    std::vector<BlockWeight> max_block_weights,
    const bool relax_max_block_weights
) {
  original_n = graph.n();
  n = graph.n();
  m = graph.m();
  original_total_node_weight = graph.total_node_weight();
  total_node_weight = graph.total_node_weight();
  total_edge_weight = graph.total_edge_weight();
  max_node_weight = graph.max_node_weight();

  k = static_cast<BlockID>(max_block_weights.size());
  _max_block_weights = std::move(max_block_weights);
  _unrelaxed_max_block_weights = _max_block_weights;
  _total_max_block_weights = std::accumulate(
      _max_block_weights.begin(), _max_block_weights.end(), static_cast<BlockWeight>(0)
  );
  _uniform_block_weights = false;

  if (relax_max_block_weights) {
    const double eps = inferred_epsilon();
    for (BlockWeight &max_block_weight : _max_block_weights) {
      max_block_weight = std::max<BlockWeight>(
          max_block_weight, std::ceil(1.0 * max_block_weight / (1.0 + eps)) + max_node_weight
      );
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
  Timer::global().print_human_readable(std::cout, false, max_timer_depth);
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

  LOG;
  LOG << "Block weights:";

  constexpr BlockID max_displayed_weights = 128;

  const int block_id_width = std::log10(std::min(max_displayed_weights, p_graph.k())) + 1;
  const int block_weight_width = std::log10(ctx.partition.original_total_node_weight) + 1;

  for (BlockID b = 0; b < std::min<BlockID>(p_graph.k(), max_displayed_weights); ++b) {
    std::stringstream ss;
    ss << "  w(" << std::left << std::setw(block_id_width) << b
       << ") = " << std::setw(block_weight_width) << p_graph.block_weight(b);
    if (p_graph.block_weight(b) > ctx.partition.max_block_weight(b)) {
      LLOG << logger::RED << ss.str() << " ";
    } else {
      LLOG << ss.str() << " ";
    }
    if ((b % 4) == 3) {
      LOG;
    }
  }
  if (p_graph.k() > max_displayed_weights) {
    LOG << "(only showing the first " << max_displayed_weights << " of " << p_graph.k()
        << " blocks)";
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
    std::span<EdgeID> xadj,
    std::span<NodeID> adjncy,
    std::span<NodeWeight> vwgt,
    std::span<EdgeWeight> adjwgt
) {
  SCOPED_HEAP_PROFILER("Borrow and mutate graph");
  SCOPED_TIMER("IO");

  const NodeID n = xadj.size() - 1;
  const EdgeID m = xadj[n];

  RECORD("nodes") StaticArray<EdgeID> nodes(n + 1, xadj.data());
  RECORD("edges") StaticArray<NodeID> edges(m, adjncy.data());
  RECORD("node_weights")
  StaticArray<NodeWeight> node_weights =
      vwgt.empty() ? StaticArray<NodeWeight>(0) : StaticArray<NodeWeight>(n, vwgt.data());
  RECORD("edge_weights")
  StaticArray<EdgeWeight> edge_weights =
      adjwgt.empty() ? StaticArray<EdgeWeight>(0) : StaticArray<EdgeWeight>(m, adjwgt.data());

  auto csr_graph = std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), false
  );
  KASSERT(shm::debug::validate_graph(*csr_graph), "invalid input graph", assert::heavy);
  set_graph(Graph(std::move(csr_graph)));
}

void KaMinPar::copy_graph(
    std::span<const EdgeID> xadj,
    std::span<const NodeID> adjncy,
    std::span<const NodeWeight> vwgt,
    std::span<const EdgeWeight> adjwgt
) {
  SCOPED_HEAP_PROFILER("Copy graph");
  SCOPED_TIMER("IO");

  const NodeID n = xadj.size() - 1;
  const EdgeID m = xadj[n];
  const bool has_node_weights = !vwgt.empty();
  const bool has_edge_weights = !adjwgt.empty();

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

shm::Graph KaMinPar::take_graph() {
  return std::move(*_graph_ptr);
}

void KaMinPar::reseed(int seed) {
  Random::reseed(seed);
}

EdgeWeight KaMinPar::compute_partition(const BlockID k, std::span<BlockID> partition) {
  return compute_partition(k, 0.03, partition);
}

EdgeWeight
KaMinPar::compute_partition(const BlockID k, const double epsilon, std::span<BlockID> partition) {
  _ctx.partition.setup(*_graph_ptr, k, epsilon);
  return compute_partition(partition);
}

EdgeWeight KaMinPar::compute_partition(
    std::vector<BlockWeight> max_block_weights, std::span<BlockID> partition
) {
  _ctx.partition.setup(*_graph_ptr, std::move(max_block_weights));
  return compute_partition(partition);
}

EdgeWeight KaMinPar::compute_partition(
    std::vector<double> max_block_weight_factors, std::span<shm::BlockID> partition
) {
  std::vector<BlockWeight> max_block_weights(max_block_weight_factors.size());
  const NodeWeight total_node_weight = _graph_ptr->total_node_weight();
  std::transform(
      max_block_weight_factors.begin(),
      max_block_weight_factors.end(),
      max_block_weights.begin(),
      [total_node_weight](const double factor) { return std::ceil(factor * total_node_weight); }
  );
  return compute_partition(std::move(max_block_weights), partition);
}

EdgeWeight KaMinPar::compute_partition(std::span<BlockID> partition) {
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
    const NodeID num_isolated_nodes = graph::count_isolated_nodes(*_graph_ptr);
    _graph_ptr->remove_isolated_nodes(num_isolated_nodes);
    _ctx.partition.n = _graph_ptr->n();
    _ctx.partition.total_node_weight = _graph_ptr->total_node_weight();

    cio::print_delimiter("Preprocessing");
    LOG << "Removed " << num_isolated_nodes << " isolated nodes";
    LOG << "  Remaining nodes:             " << _graph_ptr->n();
    LOG << "  Remaining total node weight: " << _graph_ptr->total_node_weight();
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
