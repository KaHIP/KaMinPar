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
#include <optional>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h" // IWYU pragma: export
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
    const ExecutionContext &ctx,
    const PartitionedGraph &p_graph,
    const int max_timer_depth,
    const bool parseable
) {
  const EdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const double min_imbalance = metrics::min_imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, ctx.partition);

  if (!ctx.runtime.quiet)
    cio::print_delimiter("Result Summary");

  // Statistics output that is easy to parse
  if (parseable) {
    LOG_ON(ctx.runtime) << "RESULT cut=" << cut << " imbalance=" << imbalance
                        << " feasible=" << feasible << " k=" << p_graph.k();
#ifdef KAMINPAR_ENABLE_TIMERS
    std::ostringstream timings;
    ctx.runtime.timer.print_machine_readable(timings);
    LLOG_ON(ctx.runtime) << "TIME " << timings.str();
#else  // KAMINPAR_ENABLE_TIMERS
    LOG_ON(ctx.runtime) << "TIME disabled";
#endif // KAMINPAR_ENABLE_TIMERS
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
    LOG_ON(ctx.runtime) << "MEMORY " << heap_profiler::HeapProfiler::global().peak_memory();
#else
    LOG_ON(ctx.runtime) << "MEMORY disabled";
#endif
    LOG_ON(ctx.runtime);
  }

#ifdef KAMINPAR_ENABLE_TIMERS
  std::ostringstream timings;
  ctx.runtime.timer.print_human_readable(timings, false, max_timer_depth);
  LLOG_ON(ctx.runtime) << timings.str();
#else  // KAMINPAR_ENABLE_TIMERS
  LOG_ON(ctx.runtime) << "Global Timers: disabled";
#endif // KAMINPAR_ENABLE_TIMERS
  LOG_ON(ctx.runtime);
  PRINT_HEAP_PROFILE(std::cout);
  LOG_ON(ctx.runtime) << "Partition summary:";
  if (p_graph.k() != ctx.partition.k) {
    LOG_ON(ctx.runtime) << logger::RED << "  Number of blocks: " << p_graph.k();
  } else {
    LOG_ON(ctx.runtime) << "  Number of blocks: " << p_graph.k();
  }
  LOG_ON(ctx.runtime) << "  Edge cut:         " << cut;
  LOG_ON(ctx.runtime) << "  Imbalance:        " << imbalance;
  LOG_ON(ctx.runtime) << "  Imbalance (min):  " << min_imbalance;
  if (feasible) {
    LOG_ON(ctx.runtime) << "  Feasible:         yes";
  } else {
    LOG_ON(ctx.runtime) << logger::RED << "  Feasible:         no";
  }

  LOG_ON(ctx.runtime);
  LOG_ON(ctx.runtime) << "Block weights:";

  constexpr BlockID max_displayed_weights = 128;

  const int block_id_width = std::log10(std::min(max_displayed_weights, p_graph.k())) + 1;
  const int block_weight_width = std::log10(ctx.partition.original_total_node_weight) + 1;

  for (BlockID b = 0; b < std::min<BlockID>(p_graph.k(), max_displayed_weights); ++b) {
    std::stringstream ss;
    ss << "  w(" << std::left << std::setw(block_id_width) << b
       << ") = " << std::setw(block_weight_width) << p_graph.block_weight(b);
    if (p_graph.block_weight(b) > ctx.partition.max_block_weight(b)) {
      LLOG_ON(ctx.runtime) << logger::RED << ss.str() << " ";
    } else if (p_graph.block_weight(b) < ctx.partition.min_block_weight(b)) {
      LLOG_ON(ctx.runtime) << logger::ORANGE << ss.str() << " ";
    } else {
      LLOG_ON(ctx.runtime) << ss.str() << " ";
    }
    if ((b % 4) == 3) {
      LOG_ON(ctx.runtime);
    }
  }
  if (p_graph.k() > max_displayed_weights) {
    LOG_ON(ctx.runtime) << "(only showing the first " << max_displayed_weights << " of "
                        << p_graph.k() << " blocks)";
  }
}

} // namespace

KaMinPar::KaMinPar()
    : KaMinPar(tbb::this_task_arena::max_concurrency(), create_default_context()) {}

KaMinPar::KaMinPar(const int num_threads, Context ctx)
    : _num_threads(num_threads),
      _runtime(std::make_unique<RuntimeContext>(ctx.seed, num_threads)),
      _ctx(std::move(ctx)) {}

KaMinPar::~KaMinPar() = default;
KaMinPar::KaMinPar(KaMinPar &&) noexcept = default;
KaMinPar &KaMinPar::operator=(KaMinPar &&) noexcept = default;

void KaMinPar::reseed(int seed) {
  _ctx.seed = seed;
}

int KaMinPar::get_seed() const {
  return _ctx.seed;
}

RuntimeContext &KaMinPar::runtime() {
  return *_runtime;
}
const RuntimeContext &KaMinPar::runtime() const {
  return *_runtime;
}

void KaMinPar::set_output_level(const OutputLevel output_level) {
  _output_level = output_level;
  _runtime->quiet = output_level == OutputLevel::QUIET;
}

void KaMinPar::print_timers(std::ostream &out, const int max_depth) const {
#ifdef KAMINPAR_ENABLE_TIMERS
  _runtime->timer.print_human_readable(out, false, max_depth);
#else
  (void)max_depth;
  out << "Timers disabled\n";
#endif
}

void KaMinPar::set_max_timer_depth(const int max_timer_depth) {
  _max_timer_depth = max_timer_depth;
}

Context &KaMinPar::context() {
  return _ctx;
}

const Context &KaMinPar::context() const {
  return _ctx;
}
const PartitionContext &KaMinPar::partition_context() const {
  return _partition;
}

void KaMinPar::borrow_and_mutate_graph(
    std::span<EdgeID> xadj,
    std::span<NodeID> adjncy,
    std::span<NodeWeight> vwgt,
    std::span<EdgeWeight> adjwgt
) {
  _runtime->begin_partition(_ctx.seed);
  _runtime->arena.execute([&] {
    SCOPED_HEAP_PROFILER("Borrow and mutate graph");
    SCOPED_TIMER_ON(_runtime->timer, "IO");

    const NodeID n = xadj.size() - 1;
    const EdgeID m = xadj[n];

    StaticArray<EdgeID> nodes(n + 1, xadj.data());
    StaticArray<NodeID> edges(m, adjncy.data());

    StaticArray<NodeWeight> node_weights =
        vwgt.empty() ? StaticArray<NodeWeight>(0) : StaticArray<NodeWeight>(n, vwgt.data());
    StaticArray<EdgeWeight> edge_weights =
        adjwgt.empty() ? StaticArray<EdgeWeight>(0) : StaticArray<EdgeWeight>(m, adjwgt.data());

    auto csr_graph = std::make_unique<CSRGraph>(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), false
    );
    KASSERT(shm::debug::validate_graph(*csr_graph), "invalid input graph", assert::heavy);

    set_graph(Graph(std::move(csr_graph)));
  });
}

void KaMinPar::copy_graph(
    std::span<const EdgeID> xadj,
    std::span<const NodeID> adjncy,
    std::span<const NodeWeight> vwgt,
    std::span<const EdgeWeight> adjwgt
) {
  _runtime->begin_partition(_ctx.seed);
  _runtime->arena.execute([&] {
    SCOPED_HEAP_PROFILER("Copy graph");
    SCOPED_TIMER_ON(_runtime->timer, "IO");

    const NodeID n = xadj.size() - 1;
    const EdgeID m = xadj[n];
    const bool has_node_weights = !vwgt.empty();
    const bool has_edge_weights = !adjwgt.empty();

    StaticArray<EdgeID> nodes(n + 1);
    StaticArray<NodeID> edges(m);
    StaticArray<NodeWeight> node_weights(has_node_weights ? n : 0);
    StaticArray<EdgeWeight> edge_weights(has_edge_weights ? m : 0);

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
  });
}

void KaMinPar::set_graph(Graph graph) {
  _runtime->begin_partition(_ctx.seed);
  _was_rearranged = false;
  _graph_ptr = std::make_unique<Graph>(std::move(graph));
}

const shm::Graph *KaMinPar::graph() {
  return _graph_ptr.get();
}

Graph KaMinPar::take_graph() {
  return std::move(*_graph_ptr);
}

void KaMinPar::set_k(shm::BlockID k) {
  _k = k;
}

void KaMinPar::set_uniform_max_block_weights(const double epsilon) {
  clear_max_block_weights();
  _epsilon = epsilon;
}

void KaMinPar::set_absolute_max_block_weights(
    const std::span<const shm::BlockWeight> absolute_max_block_weights
) {
  clear_max_block_weights();
  _absolute_max_block_weights.assign(
      absolute_max_block_weights.begin(), absolute_max_block_weights.end()
  );
}

void KaMinPar::set_relative_max_block_weights(
    const std::span<const double> relative_max_block_weights
) {
  clear_max_block_weights();
  _relative_max_block_weights.assign(
      relative_max_block_weights.begin(), relative_max_block_weights.end()
  );
}

void KaMinPar::clear_max_block_weights() {
  _epsilon = 0.0;
  _absolute_max_block_weights.clear();
  _relative_max_block_weights.clear();
}

void KaMinPar::set_uniform_min_block_weights(const double min_epsilon) {
  clear_min_block_weights();
  _min_epsilon = min_epsilon;
}

void KaMinPar::set_absolute_min_block_weights(
    const std::span<const shm::BlockWeight> absolute_min_block_weights
) {
  clear_min_block_weights();
  _absolute_min_block_weights.assign(
      absolute_min_block_weights.begin(), absolute_min_block_weights.end()
  );
}

void KaMinPar::set_relative_min_block_weights(
    const std::span<const double> relative_min_block_weights
) {
  clear_min_block_weights();
  _relative_min_block_weights.assign(
      relative_min_block_weights.begin(), relative_min_block_weights.end()
  );
}

void KaMinPar::clear_min_block_weights() {
  _min_epsilon = 0.0;
  _absolute_min_block_weights.clear();
  _relative_min_block_weights.clear();
}

EdgeWeight KaMinPar::compute_partition(std::span<BlockID> partition) {
  _runtime->quiet = _output_level == OutputLevel::QUIET;
  return _runtime->arena.execute([&] {
    _runtime->begin_partition(_ctx.seed);
    try {
      const auto cut = compute_partition_impl(partition);
      _runtime->finish_partition();
      return cut;
    } catch (...) {
      _runtime->reset_after_failure();
      throw;
    }
  });
}

EdgeWeight KaMinPar::compute_partition_impl(std::span<BlockID> partition) {
  const Context config = _ctx;
  ExecutionContext ctx(config, *_runtime);
  validate_partition_parameters();

  if (!ctx.runtime.quiet)
    cio::print_kaminpar_banner();
  if (!ctx.runtime.quiet)
    cio::print_build_identifier();
  if (!ctx.runtime.quiet)
    cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();
  if (!ctx.runtime.quiet)
    cio::print_delimiter("Input Summary", '#');

  KASSERT(
      (_epsilon > 0.0) + (!_absolute_max_block_weights.empty()) +
              (!_relative_max_block_weights.empty()) ==
          1,
      "unexpected state: max block weights must be configured either via epsilon, absolute or "
      "relative weights"
  );

  if (_epsilon > 0.0) {
    ctx.partition.setup(*_graph_ptr, _k, _epsilon);
  } else if (!_absolute_max_block_weights.empty()) {
    ctx.partition.setup(*_graph_ptr, _absolute_max_block_weights);
  } else {
    KASSERT(!_relative_max_block_weights.empty());

    std::vector<BlockWeight> absolute_max_block_weights(_relative_max_block_weights.size());
    const NodeWeight total_node_weight = _graph_ptr->total_node_weight();
    std::transform(
        _relative_max_block_weights.begin(),
        _relative_max_block_weights.end(),
        absolute_max_block_weights.begin(),
        [total_node_weight](const double factor) { return std::ceil(factor * total_node_weight); }
    );
    ctx.partition.setup(*_graph_ptr, std::move(absolute_max_block_weights));
  }

  KASSERT(
      (_min_epsilon > 0.0) + (!_absolute_min_block_weights.empty()) +
              (!_relative_min_block_weights.empty()) <=
          1,
      "unexpected state: min block weights must be cleared, or either configured via epsilon, "
      "absolute or relative weights"
  );

  if (_min_epsilon > 0.0) {
    ctx.partition.setup_min_block_weights(_min_epsilon);
  } else if (!_absolute_min_block_weights.empty()) {
    ctx.partition.setup_min_block_weights(_absolute_min_block_weights);
  } else if (!_relative_min_block_weights.empty()) {
    std::vector<BlockWeight> absolute_min_block_weights(_relative_min_block_weights.size());
    const NodeWeight total_node_weight = _graph_ptr->total_node_weight();
    std::transform(
        _relative_min_block_weights.begin(),
        _relative_min_block_weights.end(),
        absolute_min_block_weights.begin(),
        [total_node_weight](const double factor) { return std::ceil(factor * total_node_weight); }
    );
    ctx.partition.setup_min_block_weights(std::move(absolute_min_block_weights));
  }

  GraphCompressionStatistics compression;
  compression.enabled = config.compression.enabled;
  compression.setup(*_graph_ptr);
  ctx.parallel.num_threads = _num_threads;

  // Initialize console output
  if (_output_level >= OutputLevel::APPLICATION) {
    std::ostringstream summary;
    summary << ctx << compression;
    LLOG_ON(ctx.runtime) << summary.str();
  }

  START_HEAP_PROFILER("Partitioning");
  START_TIMER_ON(ctx.runtime.timer, "Partitioning");

  if (!_was_rearranged) {
    if (ctx.config.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
      if (ctx.config.compression.enabled) {
        LOG_WARNING_ON(ctx.runtime)
            << "A compressed graph cannot be rearranged by degree buckets. Disabling "
               "degree bucket ordering!";
      } else if (!_graph_ptr->sorted()) {
        CSRGraph &csr_graph = _graph_ptr->csr_graph();
        _graph_ptr = std::make_unique<Graph>(
            graph::rearrange_by_degree_buckets(csr_graph, ctx.runtime.timer)
        );
      }
    }

    if (ctx.config.edge_ordering == EdgeOrdering::COMPRESSION && !ctx.config.compression.enabled) {
      CSRGraph &csr_graph = _graph_ptr->csr_graph();
      graph::reorder_edges_by_compression(csr_graph, ctx.runtime.timer);
    }

    _was_rearranged = true;
  }

  std::optional<PartitionContext> original_partition_context;

  // Cut off isolated nodes if the graph has been rearranged such that the isolated nodes are placed
  // at the end.
  if (_graph_ptr->sorted()) {
    const NodeID num_isolated_nodes = graph::count_isolated_nodes(*_graph_ptr);
    if (num_isolated_nodes > 0 && ctx.partition.has_min_block_weights()) {
      original_partition_context = ctx.partition;
    }
    reified(*_graph_ptr, [&](auto &graph) {
      graph.remove_isolated_nodes(num_isolated_nodes);
      ctx.partition.n = graph.n();
      ctx.partition.total_node_weight = graph.total_node_weight();
    });
    if (original_partition_context) {
      const BlockWeight isolated_weight =
          original_partition_context->total_node_weight - ctx.partition.total_node_weight;
      ctx.partition.setup_min_block_weights(
          graph::compute_nonisolated_min_block_weights(*original_partition_context, isolated_weight)
      );
    }

    if (!ctx.runtime.quiet)
      cio::print_delimiter("Preprocessing");
    LOG_ON(ctx.runtime) << "Removed " << num_isolated_nodes << " isolated nodes";
    LOG_ON(ctx.runtime) << "  Remaining nodes:             " << _graph_ptr->n();
    LOG_ON(ctx.runtime) << "  Remaining total node weight: " << _graph_ptr->total_node_weight();
  }

  // Perform actual partitioning
  PartitionedGraph p_graph = [&] {
    auto partitioner = factory::create_partitioner(*_graph_ptr, ctx);
    if (_output_level >= OutputLevel::DEBUG) {
      partitioner->enable_metrics_output();
    }
    PartitionedGraph p_graph = partitioner->partition();

    START_TIMER_ON(ctx.runtime.timer, "Deallocation");
    partitioner.reset();
    STOP_TIMER_ON(ctx.runtime.timer);

    return p_graph;
  }();

  // Re-integrate isolated nodes that were cut off during preprocessing
  if (_graph_ptr->sorted()) {
    SCOPED_HEAP_PROFILER("Re-integrate isolated nodes");
    SCOPED_TIMER_ON(ctx.runtime.timer, "Re-integrate isolated nodes");

    if (original_partition_context) {
      ctx.partition = std::move(*original_partition_context);
    }

    reified(*_graph_ptr, [&](auto &graph) {
      const NodeID num_isolated_nodes = graph.integrate_isolated_nodes();
      p_graph = graph::assign_isolated_nodes(std::move(p_graph), num_isolated_nodes, ctx.partition);
    });
  }

  STOP_TIMER_ON(ctx.runtime.timer);
  STOP_HEAP_PROFILER();

  START_TIMER_ON(ctx.runtime.timer, "IO");
  reified(*_graph_ptr, [&](const auto &graph) {
    if (graph.permuted()) {
      tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID u) {
        partition[u] = p_graph.block(graph.map_original_node(u));
      });
    } else {
      tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID u) {
        partition[u] = p_graph.block(u);
      });
    }
  });
  STOP_TIMER_ON(ctx.runtime.timer);

  // Print some statistics
  STOP_TIMER_ON(ctx.runtime.timer); // stop root timer
  if (_output_level >= OutputLevel::APPLICATION) {
    print_statistics(ctx, p_graph, _max_timer_depth, _output_level >= OutputLevel::EXPERIMENT);
  }

  const EdgeWeight final_cut = metrics::edge_cut(p_graph);

#ifdef KAMINPAR_ENABLE_TIMERS
  // Keep the completed timer tree available through runtime().timer; reset on the next call.
#endif // KAMINPAR_ENABLE_TIMERS

  _partition = ctx.partition;
  return final_cut;
}

void KaMinPar::validate_partition_parameters() {
  if (_graph_ptr == nullptr) {
    throw std::invalid_argument(
        "Call KaMinPar::borrow_and_mutate_graph() or KaMinPar::copy_graph() before calling "
        "KaMinPar::compute_partition()."
    );
  }

  if (_k == 0) {
    throw std::invalid_argument(
        "Call KaMinPar::set_k() before calling KaMinPar::compute_partition()."
    );
  }

  if (_epsilon == 0.0 && _absolute_max_block_weights.empty() &&
      _relative_max_block_weights.empty()) {
    throw std::invalid_argument(
        "Call KaMinPar::set_uniform_max_block_weights(), "
        "KaMinPar::set_absolute_max_block_weights() or KaMinPar::set_relative_max_block_weights() "
        "before calling KaMinPar::compute_partition()."
    );
  }
  if (!_absolute_max_block_weights.empty() &&
      _absolute_max_block_weights.size() != static_cast<std::size_t>(_k)) {
    throw std::invalid_argument(
        "Length of the span passed to KaMinPar::set_absolute_max_block_weights() does not match "
        "the number of blocks passed to KaMinPar::set_k()."
    );
  }
  if (!_relative_max_block_weights.empty() &&
      _relative_max_block_weights.size() != static_cast<std::size_t>(_k)) {
    throw std::invalid_argument(
        "Length of the span passed to KaMinPar::set_relative_max_block_weights() does not match "
        "the number of blocks passed to KaMinPar::set_k()."
    );
  }

  if (!_absolute_min_block_weights.empty() &&
      _absolute_min_block_weights.size() != static_cast<std::size_t>(_k)) {
    throw std::invalid_argument(
        "Length of the span passed to KaMinPar::set_absolute_min_block_weights() does not match "
        "the number of blocks passed to KaMinPar::set_k()."
    );
  }
  if (!_relative_min_block_weights.empty() &&
      _relative_min_block_weights.size() != static_cast<std::size_t>(_k)) {
    throw std::invalid_argument(
        "Length of the span passed to KaMinPar::set_relative_min_block_weights() does not match "
        "the number of blocks passed to KaMinPar::set_k()."
    );
  }
}

EdgeWeight KaMinPar::compute_partition(const BlockID k, std::span<BlockID> partition) {
  set_k(k);
  set_uniform_max_block_weights(kDefaultEpsilon);
  return compute_partition(partition);
}

EdgeWeight
KaMinPar::compute_partition(const BlockID k, const double epsilon, std::span<BlockID> partition) {
  set_k(k);
  set_uniform_max_block_weights(epsilon);
  return compute_partition(partition);
}

EdgeWeight KaMinPar::compute_partition(
    std::vector<BlockWeight> max_block_weights, std::span<BlockID> partition
) {
  set_k(static_cast<BlockID>(max_block_weights.size()));
  set_absolute_max_block_weights(max_block_weights);
  return compute_partition(partition);
}

EdgeWeight KaMinPar::compute_partition(
    std::vector<double> max_block_weight_factors, std::span<shm::BlockID> partition
) {
  set_k(static_cast<BlockID>(max_block_weight_factors.size()));
  set_relative_max_block_weights(max_block_weight_factors);
  return compute_partition(partition);
}

} // namespace kaminpar
