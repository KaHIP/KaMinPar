/*******************************************************************************
 * @file:   balanced_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   12.06.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/balanced_coarsener.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
BalancedCoarsener::BalancedCoarsener(const Context &ctx, const PartitionContext &p_ctx)
    : _clustering_algorithm(factory::create_clusterer(ctx)),
      _c_ctx(ctx.coarsening),
      _p_ctx(p_ctx) {}

void BalancedCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

bool BalancedCoarsener::coarsen() {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  START_HEAP_PROFILER("Allocation");
  RECORD("clustering") StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const Graph &graph = current();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeWeight total_node_weight = graph.total_node_weight();
  const NodeID prev_n = graph.n();

  const bool was_quiet = Logger::is_quiet();
  Logger::set_quiet_mode(true);
  START_TIMER("Partitioning for Clustering");
  DISABLE_TIMERS();
  {
    const NodeWeight max_cluster_weight =
        compute_max_cluster_weight<NodeWeight>(_c_ctx, _p_ctx, prev_n, total_node_weight);

    Context ctx = create_largek_fast_context();
    ctx.partition.epsilon = _c_ctx.clustering.max_allowed_imbalance;
    ctx.partition.k = total_node_weight / max_cluster_weight;
    ctx.partition.setup(graph);

    DeepMultilevelPartitioner partitioner(graph, ctx);
    PartitionedGraph p_graph = partitioner.partition();
    p_graph.pfor_nodes([&](const NodeID u) { clustering[u] = p_graph.block(u); });
  }
  ENABLE_TIMERS();
  STOP_TIMER();
  Logger::set_quiet_mode(was_quiet);

  START_HEAP_PROFILER("Contract graph");
  auto coarsened = TIMED_SCOPE("Contract graph") {
    return contract_clustering(
        current(), std::move(clustering), _c_ctx.contraction, _contraction_m_ctx
    );
  };
  _hierarchy.push_back(std::move(coarsened));
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

PartitionedGraph BalancedCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
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

std::unique_ptr<CoarseGraph> BalancedCoarsener::pop_hierarchy(PartitionedGraph &&p_graph) {
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

bool BalancedCoarsener::keep_allocated_memory() const {
  return level() >= _c_ctx.clustering.max_mem_free_coarsening_level;
}
} // namespace kaminpar::shm
