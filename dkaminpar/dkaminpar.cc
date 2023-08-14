/*******************************************************************************
 * Public interface of the distributed partitioner.
 *
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   30.01.2023
 ******************************************************************************/
#include "dkaminpar/dkaminpar.h"

#include <utility>

#include <mpi.h>
#include <omp.h>
#include <tbb/global_control.h>
#include <tbb/parallel_invoke.h>

#include "dkaminpar/context_io.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/ghost_node_mapper.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/rearrangement.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/timer.h"

#include "kaminpar/context.h"

#include "common/console_io.h"
#include "common/environment.h"
#include "common/random.h"

namespace kaminpar {
using namespace dist;

namespace {
void print_partition_summary(
    const Context &ctx,
    const DistributedPartitionedGraph &p_graph,
    const int max_timer_depth,
    const bool parseable,
    const bool root
) {
  const auto edge_cut = metrics::edge_cut(p_graph);
  const auto imbalance = metrics::imbalance(p_graph);
  const auto feasible =
      metrics::is_feasible(p_graph, ctx.partition) && p_graph.k() == ctx.partition.k;

  if (!root) {
    // Non-root PEs are only needed to compute the partition metrics
    return;
  }

  cio::print_delimiter("Result Summary");

  if (parseable) {
    LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible
        << " k=" << p_graph.k();
    std::cout << "TIME ";
    Timer::global().print_machine_readable(std::cout);
  }

  Timer::global().print_human_readable(std::cout, max_timer_depth);
  LOG;
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
    LOG << "INPUT_GRAPH "
        << "global_n=" << graph.global_n() << " "
        << "global_m=" << graph.global_m() << " "
        << "n=[" << n_str << "] "
        << "m=[" << m_str << "] "
        << "ghost_n=[" << ghost_n_str << "]";
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

dKaMinPar::dKaMinPar(MPI_Comm comm, const int num_threads, const Context ctx)
    : _comm(comm),
      _num_threads(num_threads),
      _ctx(ctx),
      _gc(tbb::global_control::max_allowed_parallelism, num_threads) {
  omp_set_num_threads(num_threads);
  Random::seed = 0;
}

dKaMinPar::~dKaMinPar() = default;

void dKaMinPar::set_output_level(const OutputLevel output_level) {
  _output_level = output_level;
}

void dKaMinPar::set_max_timer_depth(const int max_timer_depth) {
  _max_timer_depth = max_timer_depth;
}

Context &dKaMinPar::context() {
  return _ctx;
}

void dKaMinPar::import_graph(
    GlobalNodeID *vtxdist,
    GlobalEdgeID *xadj,
    GlobalNodeID *adjncy,
    GlobalNodeWeight *vwgt,
    GlobalEdgeWeight *adjwgt
) {
  SCOPED_TIMER("IO");

  const PEID size = mpi::get_comm_size(_comm);
  const PEID rank = mpi::get_comm_rank(_comm);

  const NodeID n = static_cast<NodeID>(vtxdist[rank + 1] - vtxdist[rank]);
  const GlobalNodeID from = vtxdist[rank];
  const GlobalNodeID to = vtxdist[rank + 1];
  const EdgeID m = static_cast<EdgeID>(xadj[n]);

  StaticArray<GlobalNodeID> node_distribution(vtxdist, vtxdist + size + 1);
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
        if (vwgt == nullptr) {
          return;
        }
        node_weights.resize(n);
        tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = vwgt[u]; });
      },
      [&] {
        if (adjwgt == nullptr) {
          return;
        }
        edge_weights.resize(m);
        tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) { edge_weights[e] = adjwgt[e]; });
      }
  );

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();

  _graph_ptr = std::make_unique<DistributedGraph>(
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
  );
}

GlobalEdgeWeight dKaMinPar::compute_partition(const int seed, const BlockID k, BlockID *partition) {
  auto &graph = *_graph_ptr;

  const PEID size = mpi::get_comm_size(_comm);
  const PEID rank = mpi::get_comm_rank(_comm);
  const bool root = rank == 0;

  KASSERT(graph::debug::validate(graph), "input graph failed graph verification", assert::heavy);

  // Make number of processes and number of threads available via
  // ParallelContext
  _ctx.parallel.num_mpis = size;
  _ctx.parallel.num_threads = _num_threads;
  _ctx.initial_partitioning.kaminpar.parallel.num_threads = _ctx.parallel.num_threads;
  _ctx.partition.k = k;
  _ctx.partition.graph = std::make_unique<GraphContext>(graph, _ctx.partition);

  // Initialize PRNG and console output
  Random::seed = seed;
  Logger::set_quiet_mode(_output_level == OutputLevel::QUIET);

  if (_output_level >= OutputLevel::APPLICATION) {
    print_input_summary(_ctx, graph, _output_level == OutputLevel::EXPERIMENT, root);
  }

  START_TIMER("Partitioning");
  if (!_was_rearranged) {
    graph = graph::rearrange(std::move(graph), _ctx);
    _was_rearranged = true;
  }
  auto p_graph = factory::create_partitioner(_ctx, graph)->partition();
  STOP_TIMER();

  KASSERT(
      graph::debug::validate_partition(p_graph),
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
  STOP_TIMER(); // stop root timer

  if (_output_level >= OutputLevel::APPLICATION) {
    print_partition_summary(
        _ctx, p_graph, _max_timer_depth, _output_level == OutputLevel::EXPERIMENT, root
    );
  }

  return metrics::edge_cut(p_graph);
}
} // namespace kaminpar
