/*******************************************************************************
 * Public interface of the distributed partitioner.
 *
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   30.01.2023
 ******************************************************************************/
#include "kaminpar-dist/dkaminpar.h"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>

#include <mpi.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/context_io.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/rearrangement.h"
#include "kaminpar-dist/graphutils/synchronization.h"
#include "kaminpar-dist/heap_profiler.h"
#include "kaminpar-dist/logger.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/random.h"

namespace kaminpar {

namespace dist {

void PartitionContext::setup(
    const AbstractDistributedGraph &graph,
    const BlockID k,
    const double epsilon,
    const bool relax_max_block_weights
) {
  _epsilon = epsilon;

  // this->global_total_node_weight not yet initialized: use graph.total_node_weight instead
  const BlockWeight perfectly_balanced_block_weight =
      std::ceil(1.0 * graph.global_total_node_weight() / k);
  std::vector<BlockWeight> max_block_weights(k, (1.0 + epsilon) * perfectly_balanced_block_weight);
  setup(graph, std::move(max_block_weights), relax_max_block_weights);

  _uniform_block_weights = true;
}

void PartitionContext::setup(
    const AbstractDistributedGraph &graph,
    std::vector<BlockWeight> max_block_weights,
    const bool relax_max_block_weights
) {
  global_n = graph.global_n();
  n = graph.n();
  total_n = graph.total_n();
  global_m = graph.global_m();
  m = graph.m();
  global_total_node_weight = graph.global_total_node_weight();
  total_node_weight = graph.total_node_weight();
  global_max_node_weight = graph.global_max_node_weight();
  global_total_edge_weight = graph.global_total_edge_weight();
  total_edge_weight = graph.total_edge_weight();

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
          max_block_weight, std::ceil(1.0 * max_block_weight / (1.0 + eps)) + global_max_node_weight
      );
    }
  }
}

int ChunksContext::compute(const ParallelContext &parallel) const {
  if (fixed_num_chunks > 0) {
    return fixed_num_chunks;
  }
  const PEID num_pes =
      scale_chunks_with_threads ? parallel.num_threads * parallel.num_mpis : parallel.num_mpis;
  return std::max<std::size_t>(min_num_chunks, total_num_chunks / num_pes);
}

bool LabelPropagationCoarseningContext::should_merge_nonadjacent_clusters(
    const NodeID old_n, const NodeID new_n
) const {
  return (1.0 - 1.0 * new_n / old_n) <= merge_nonadjacent_clusters_threshold;
}

bool RefinementContext::includes_algorithm(const RefinementAlgorithm algorithm) const {
  return std::find(algorithms.begin(), algorithms.end(), algorithm) != algorithms.end();
}

void GraphCompressionContext::setup(const DistributedCompressedGraph &graph) {
  constexpr int kRoot = 0;
  const MPI_Comm comm = graph.communicator();
  const int rank = mpi::get_comm_rank(comm);

  compressed_graph_sizes =
      mpi::gather<std::size_t, std::vector<std::size_t>>(graph.memory_space(), kRoot, comm);
  uncompressed_graph_sizes = mpi::gather<std::size_t, std::vector<std::size_t>>(
      graph.uncompressed_memory_space(), kRoot, comm
  );
  num_nodes = mpi::gather<NodeID, std::vector<NodeID>>(graph.n(), kRoot, comm);
  num_edges = mpi::gather<EdgeID, std::vector<EdgeID>>(graph.m(), kRoot, comm);

  const auto compression_ratios = mpi::gather(graph.compression_ratio(), kRoot, comm);
  if (rank == kRoot) {
    const auto size = static_cast<double>(compression_ratios.size());
    avg_compression_ratio =
        std::reduce(compression_ratios.begin(), compression_ratios.end()) / size;
    min_compression_ratio = *std::min_element(compression_ratios.begin(), compression_ratios.end());
    max_compression_ratio = *std::max_element(compression_ratios.begin(), compression_ratios.end());

    const auto largest_compressed_graph_it =
        std::max_element(compressed_graph_sizes.begin(), compressed_graph_sizes.end());
    largest_compressed_graph = *largest_compressed_graph_it;

    const auto largest_compressed_graph_rank =
        std::distance(compressed_graph_sizes.begin(), largest_compressed_graph_it);
    largest_compressed_graph_prev_size =
        largest_compressed_graph * compression_ratios[largest_compressed_graph_rank];

    const auto largest_uncompressed_graph_it =
        std::max_element(uncompressed_graph_sizes.begin(), uncompressed_graph_sizes.end());
    largest_uncompressed_graph = *largest_uncompressed_graph_it;

    const auto largest_uncompressed_graph_rank =
        std::distance(uncompressed_graph_sizes.begin(), largest_uncompressed_graph_it);
    largest_uncompressed_graph_after_size =
        largest_uncompressed_graph / compression_ratios[largest_uncompressed_graph_rank];
  }
}

namespace {

void print_partition_summary(
    const Context &ctx,
    const DistributedPartitionedGraph &p_graph,
    const int max_timer_depth,
    const bool parseable,
    const bool root
) {
  MPI_Comm comm = p_graph.communicator();

  const auto edge_cut = metrics::edge_cut(p_graph);
  const auto imbalance = metrics::imbalance(p_graph);
  const auto feasible =
      metrics::is_feasible(p_graph, ctx.partition) && p_graph.k() == ctx.partition.k;

#ifdef KAMINPAR_ENABLE_TIMERS
  finalize_distributed_timer(Timer::global(), comm);
#endif // KAMINPAR_ENABLE_TIMERS

  int heap_profile_root_rank;
  if constexpr (kHeapProfiling) {
    auto &heap_profiler = heap_profiler::HeapProfiler::global();
    heap_profile_root_rank = finalize_distributed_heap_profiler(heap_profiler, comm);
  }

  if (root) {
    cio::print_delimiter("Result Summary");

    if (parseable) {
      LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible
          << " k=" << p_graph.k();
#ifdef KAMINPAR_ENABLE_TIMERS
      std::cout << "TIME ";
      Timer::global().print_machine_readable(std::cout);
#else  // KAMINPAR_ENABLE_TIMERS
      LOG << "TIME disabled";
#endif // KAMINPAR_ENABLE_TIMERS
    }

#ifdef KAMINPAR_ENABLE_TIMERS
    Timer::global().print_human_readable(std::cout, true, max_timer_depth);
#else  // KAMINPAR_ENABLE_TIMERS
    LOG << "Global Timers: disabled";
#endif // KAMINPAR_ENABLE_TIMERS
    LOG;
  }

  if constexpr (kHeapProfiling) {
    SingleSynchronizedLogger logger(heap_profile_root_rank);

    const bool heap_profile_root = heap_profile_root_rank == mpi::get_comm_rank(comm);
    if (heap_profile_root) {
      PRINT_HEAP_PROFILE(logger.output());
    }
  }

  if (root) {
    LOG << "Partition summary:";
    if (p_graph.k() != ctx.partition.k) {
      LOG << logger::RED << "  Number of blocks: " << p_graph.k();
    } else {
      LOG << "  Number of blocks: " << p_graph.k();
    }
    LOG << "  Edge cut:         " << edge_cut;
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
    const int block_weight_width = std::log10(ctx.partition.global_total_node_weight) + 1;

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
}

void print_input_summary(
    const Context &ctx, const DistributedGraph &graph, const bool parseable, const bool root
) {
  const auto n_str = mpi::gather_statistics_str<GlobalNodeID>(graph.n(), MPI_COMM_WORLD);
  const auto m_str = mpi::gather_statistics_str<GlobalEdgeID>(graph.m(), MPI_COMM_WORLD);
  const auto ghost_n_str =
      mpi::gather_statistics_str<GlobalNodeID>(graph.ghost_n(), MPI_COMM_WORLD);

  if (root && parseable) {
    LOG << "EXECUTION_MODE num_mpis=" << ctx.parallel.num_mpis
        << " num_threads=" << ctx.parallel.num_threads;
    LOG << "INPUT_GRAPH " << "global_n=" << graph.global_n() << " "
        << "global_m=" << graph.global_m() << " " << "n=[" << n_str << "] " << "m=[" << m_str
        << "] " << "ghost_n=[" << ghost_n_str << "]";
  }

  // Output
  if (root) {
    cio::print_dkaminpar_banner();
    cio::print_build_identifier();
    cio::print_build_datatypes<
        NodeID,
        EdgeID,
        NodeWeight,
        EdgeWeight,
        shm::NodeWeight,
        shm::EdgeWeight>();
    cio::print_delimiter("Input Summary");
    LOG << "Execution mode:               " << ctx.parallel.num_mpis << " MPI process"
        << (ctx.parallel.num_mpis > 1 ? "es" : "") << " a " << ctx.parallel.num_threads << " thread"
        << (ctx.parallel.num_threads > 1 ? "s" : "");
  }
  print(ctx, root, std::cout, graph.communicator());
  if (root) {
    cio::print_delimiter("Partitioning");
  }
}

} // namespace

} // namespace dist

using namespace dist;

dKaMinPar::dKaMinPar(MPI_Comm comm, const int num_threads, const Context ctx)
    : _comm(comm),
      _num_threads(num_threads),
      _ctx(ctx),
      _gc(tbb::global_control::max_allowed_parallelism, num_threads) {
#ifdef KAMINPAR_ENABLE_TIMERS
  GLOBAL_TIMER.reset();
#endif // KAMINPAR_ENABLE_TIMERS
}

dKaMinPar::~dKaMinPar() = default;

void dKaMinPar::reseed(const int seed) {
  Random::reseed(seed);
}

void dKaMinPar::set_output_level(const OutputLevel output_level) {
  _output_level = output_level;
}

void dKaMinPar::set_max_timer_depth(const int max_timer_depth) {
  _max_timer_depth = max_timer_depth;
}

Context &dKaMinPar::context() {
  return _ctx;
}

void dKaMinPar::copy_graph(
    const std::span<const GlobalNodeID> vtxdist,
    const std::span<const GlobalEdgeID> xadj,
    const std::span<const GlobalNodeID> adjncy,
    const std::span<const GlobalNodeWeight> vwgt,
    const std::span<const GlobalEdgeWeight> adjwgt
) {
  SCOPED_TIMER("Import graph");

  const PEID size = mpi::get_comm_size(_comm);
  const PEID rank = mpi::get_comm_rank(_comm);

  const NodeID n = static_cast<NodeID>(vtxdist[rank + 1] - vtxdist[rank]);
  const GlobalNodeID from = vtxdist[rank];
  const GlobalNodeID to = vtxdist[rank + 1];
  const EdgeID m = static_cast<EdgeID>(xadj[n]);

  StaticArray<GlobalNodeID> node_distribution(vtxdist.begin(), vtxdist.begin() + size + 1);
  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = m;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      _comm
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  // We copy the graph to change the data types to 32 bits. If we do not want 32 bit IDs, we could
  // use the original memory directly; @todo
  // Except for the node weights, which must be re-allocated to make room for the weights of ghost
  // nodes
  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
  graph::GhostNodeMapper mapper(rank, node_distribution);

  tbb::parallel_invoke(
      [&] {
        nodes.resize(n + 1);
        tbb::parallel_for<NodeID>(0, n + 1, [&](const NodeID u) { nodes[u] = xadj[u]; });
      },
      [&] {
        edges.resize(m);
        tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
          const GlobalNodeID v = adjncy[e];
          if (v >= from && v < to) {
            edges[e] = static_cast<NodeID>(v - from);
          } else {
            edges[e] = mapper.new_ghost_node(v);
          }
        });
      },
      [&] {
        if (adjwgt.empty()) {
          return;
        }
        edge_weights.resize(m);
        tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) { edge_weights[e] = adjwgt[e]; });
      }
  );

  if (!vwgt.empty()) {
    // Allocate enough room for ghost nodes; we fill them in after constructing the graph
    node_weights.resize(n + mapper.next_ghost_node());
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = vwgt[u]; });
  }

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();

  set_graph({std::make_unique<DistributedCSRGraph>(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      std::move(ghost_owner),
      std::move(ghost_to_global),
      std::move(global_to_ghost),
      false,
      _comm
  )});

  // Fill in ghost node weights
  bool has_vwgt = !vwgt.empty();
  MPI_Allreduce(MPI_IN_PLACE, &has_vwgt, 1, MPI_C_BOOL, MPI_LOR, _comm);
  if (has_vwgt) {
    graph::synchronize_ghost_node_weights(*_graph_ptr);
  }
}

void dKaMinPar::set_graph(DistributedGraph graph) {
  _was_rearranged = false;
  _graph_ptr = std::make_unique<DistributedGraph>(std::move(graph));
}

const DistributedGraph *dKaMinPar::graph() const {
  return _graph_ptr.get();
}

GlobalEdgeWeight dKaMinPar::compute_partition(const BlockID k, const std::span<BlockID> partition) {
  return compute_partition(k, 0.03, partition);
}

GlobalEdgeWeight dKaMinPar::compute_partition(
    const BlockID k, const double epsilon, const std::span<BlockID> partition
) {
  _ctx.partition.setup(*_graph_ptr, k, epsilon);
  return compute_partition(partition);
}

GlobalEdgeWeight dKaMinPar::compute_partition(
    std::vector<BlockWeight> max_block_weights, const std::span<BlockID> partition
) {
  _ctx.partition.setup(*_graph_ptr, std::move(max_block_weights));
  return compute_partition(partition);
}

GlobalEdgeWeight dKaMinPar::compute_partition(
    std::vector<double> max_block_weight_factors, const std::span<dist::BlockID> partition
) {
  std::vector<BlockWeight> max_block_weights(max_block_weight_factors.size());
  const GlobalNodeWeight total_node_weight = _graph_ptr->global_total_node_weight();
  std::transform(
      max_block_weight_factors.begin(),
      max_block_weight_factors.end(),
      max_block_weights.begin(),
      [total_node_weight](const double factor) { return std::ceil(factor * total_node_weight); }
  );
  return compute_partition(std::move(max_block_weights), partition);
}

GlobalEdgeWeight dKaMinPar::compute_partition(const std::span<BlockID> partition) {
  DistributedGraph &graph = *_graph_ptr;

  const PEID size = mpi::get_comm_size(_comm);
  const PEID rank = mpi::get_comm_rank(_comm);
  const bool root = rank == 0;

  // @todo: discuss: should there be an option to enable this check independently from the assertion
  // level?
  // The binary interface already implements graph validation via KaGen, which can be enabled as a
  // CLI flag. There is no such option when using the library interface.
  KASSERT(
      dist::debug::validate_graph(graph), "input graph failed graph verification", assert::heavy
  );

  // Setup the remaining context options that are passed in via the constructor
  _ctx.parallel.num_mpis = size;
  _ctx.parallel.num_threads = _num_threads;
  _ctx.initial_partitioning.kaminpar.parallel.num_threads = _ctx.parallel.num_threads;
  if (_ctx.compression.enabled) {
    _ctx.compression.setup(_graph_ptr->compressed_graph());
  }

  // Initialize console output
  Logger::set_quiet_mode(_output_level == OutputLevel::QUIET);
  if (_output_level >= OutputLevel::APPLICATION) {
    print_input_summary(_ctx, graph, _output_level >= OutputLevel::EXPERIMENT, root);
  }

  START_HEAP_PROFILER("Partitioning");
  START_TIMER("Partitioning");
  if (!_was_rearranged && _ctx.rearrange_by != GraphOrdering::NATURAL) {
    if (_ctx.compression.enabled) {
      LOG_WARNING_ROOT << "A compressed graph cannot be rearranged by degree buckets. Disabling "
                          "degree bucket ordering!";
      _ctx.rearrange_by = GraphOrdering::NATURAL;
    } else {
      START_HEAP_PROFILER("Rearrange input graph");
      DistributedCSRGraph &csr_graph = _graph_ptr->csr_graph();
      graph = DistributedGraph(
          std::make_unique<DistributedCSRGraph>(graph::rearrange(std::move(csr_graph), _ctx))
      );
      STOP_HEAP_PROFILER();
    }

    _was_rearranged = true;
  }
  auto p_graph = [&] {
    auto partitioner = factory::create_partitioner(_ctx, graph);
    if (_output_level >= OutputLevel::DEBUG) {
      partitioner->enable_graph_stats_output();
    }
    return partitioner->partition();
  }();
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  KASSERT(
      dist::debug::validate_partition(p_graph),
      "graph partition verification failed after partitioning",
      assert::heavy
  );

  START_TIMER("IO");
  if (graph.permuted()) {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(graph.map_original_node(u));
    });
  } else {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(u);
    });
  }
  STOP_TIMER();

  mpi::barrier(MPI_COMM_WORLD);

  // Print some statistics
  STOP_TIMER(); // stop root timer
  if (_output_level >= OutputLevel::APPLICATION) {
    print_partition_summary(
        _ctx, p_graph, _max_timer_depth, _output_level >= OutputLevel::EXPERIMENT, root
    );
  }

  const EdgeWeight final_cut = metrics::edge_cut(p_graph);

#ifdef KAMINPAR_ENABLE_TIMERS
  GLOBAL_TIMER.reset();
#endif // KAMINPAR_ENABLE_TIMERS

  return final_cut;
}

} // namespace kaminpar
