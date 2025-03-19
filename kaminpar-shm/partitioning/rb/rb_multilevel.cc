/*******************************************************************************
 * Partitioning scheme that uses toplevel multilevel recursvie bipartitioning.
 *
 * @file:   rb_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/rb/rb_multilevel.h"

#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/partitioning/helper.h"

#include "kaminpar-common/math.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

RBMultilevelPartitioner::RBMultilevelPartitioner(const Graph &graph, const Context &ctx)
    : _input_graph(graph),
      _input_ctx(ctx),
      _bipartitioner_pool(_input_ctx) {
  if (!math::is_power_of_2(_input_ctx.partition.k)) {
    throw std::invalid_argument("k must be a power of two");
  }
}

PartitionedGraph RBMultilevelPartitioner::partition() {
  DISABLE_TIMERS();
  PartitionedGraph p_graph = [&] {
    if (_input_ctx.partitioning.rb_switch_to_seq_factor > 0) {
      const BlockID parallel_k = math::ceil2(static_cast<std::uint32_t>(
          _input_ctx.partitioning.rb_switch_to_seq_factor * _input_ctx.parallel.num_threads
      ));

      PartitionedGraph p_graph = partition_recursive(
          _input_graph, 0, 1, std::min<BlockID>(_input_ctx.partition.k, parallel_k)
      );

      if (parallel_k < _input_ctx.partition.k) {
        graph::SubgraphMemory _extraction_mem_pool_ets(
            p_graph.graph().n(), _input_ctx.partition.k, p_graph.graph().m()
        );
        partitioning::TemporarySubgraphMemoryEts _tmp_extraction_mem_pool_ets;
        InitialBipartitionerWorkerPool _bipartitioner_pool(_input_ctx);

        partitioning::extend_partition(
            p_graph,
            _input_ctx.partition.k,
            _input_ctx,
            _extraction_mem_pool_ets,
            _tmp_extraction_mem_pool_ets,
            _bipartitioner_pool,
            _input_ctx.parallel.num_threads
        );
      }

      return p_graph;
    } else {
      return partition_recursive(_input_graph, 0, 1, _input_ctx.partition.k);
    }
  }();
  ENABLE_TIMERS();

  if (_input_ctx.partitioning.rb_enable_kway_toplevel_refinement) {
    SCOPED_TIMER("Toplevel Refinement");
    auto refiner = factory::create_refiner(_input_ctx);
    refiner->initialize(p_graph);
    refiner->refine(p_graph, _input_ctx.partition);
  }

  return p_graph;
}

PartitionedGraph RBMultilevelPartitioner::partition_recursive(
    const Graph &graph,
    const BlockID current_block,
    const BlockID current_k,
    const BlockID desired_k
) {
  DBG << "Partitioning subgraphs " << current_block << " of " << current_k;
  auto p_graph = bipartition(graph, current_block, current_k);

  if (current_k * 2 < desired_k) {
    graph::SubgraphMemory memory(p_graph);
    const auto extraction = extract_subgraphs(p_graph, p_graph.k(), memory);
    const auto &subgraphs = extraction.subgraphs;
    const auto &mapping = extraction.node_mapping;

    PartitionedGraph p_graph1, p_graph2;
    tbb::parallel_invoke(
        [&] {
          p_graph1 = partition_recursive(subgraphs[0], 2 * current_block, current_k * 2, desired_k);
        },
        [&] {
          p_graph2 =
              partition_recursive(subgraphs[1], 2 * current_block + 1, current_k * 2, desired_k);
        }
    );

    ScalableVector<StaticArray<BlockID>> subgraph_partitions(2);
    subgraph_partitions[0] = p_graph1.take_raw_partition();
    subgraph_partitions[1] = p_graph2.take_raw_partition();

    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph),
        subgraph_partitions,
        p_graph1.k() + p_graph2.k(),
        _input_ctx.partition.k,
        mapping
    );
  }

  return p_graph;
}

PartitionedGraph RBMultilevelPartitioner::bipartition(
    const Graph &graph, const BlockID current_block, const BlockID current_k
) {
  // set k to 2 for max cluster weight computation
  PartitionContext p_ctx =
      partitioning::create_twoway_context(_input_ctx, current_block, current_k, graph);

  auto coarsener = factory::create_coarsener(_input_ctx, p_ctx);
  coarsener->initialize(&graph);

  // Coarsening
  const Graph *c_graph = &graph;
  bool shrunk = true;
  while (shrunk && c_graph->n() > 2 * _input_ctx.coarsening.contraction_limit) {
    shrunk = coarsener->coarsen();
    c_graph = &coarsener->current();
  }

  // Initial bipartitioning
  PartitionedGraph p_graph =
      _bipartitioner_pool.bipartition(c_graph, current_block, current_k, true);

  // Uncoarsening + Refinement
  auto refiner = factory::create_refiner(_input_ctx);

  while (!coarsener->empty()) {
    refiner->initialize(p_graph);
    refiner->refine(p_graph, p_ctx);
    p_graph = coarsener->uncoarsen(std::move(p_graph));
  }

  refiner->initialize(p_graph);
  refiner->refine(p_graph, p_ctx);

  return p_graph;
}

} // namespace kaminpar::shm
