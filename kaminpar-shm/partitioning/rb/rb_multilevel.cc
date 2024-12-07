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

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

RBMultilevelPartitioner::RBMultilevelPartitioner(const Graph &graph, const Context &ctx)
    : _input_graph(graph),
      _input_ctx(ctx),
      _bipartitioner_pool(_input_ctx) {}

PartitionedGraph RBMultilevelPartitioner::partition() {
  DISABLE_TIMERS();
  PartitionedGraph p_graph = partition_recursive(_input_graph, 0, 1);
  ENABLE_TIMERS();
  return p_graph;
}

PartitionedGraph RBMultilevelPartitioner::partition_recursive(
    const Graph &graph, const BlockID current_block, const BlockID current_k
) {
  auto p_graph = bipartition(graph, current_block, current_k);

  if (current_k < _input_ctx.partition.k) {
    graph::SubgraphMemory memory(p_graph);
    const auto extraction = extract_subgraphs(p_graph, _input_ctx.partition.k, memory);
    const auto &subgraphs = extraction.subgraphs;
    const auto &mapping = extraction.node_mapping;

    PartitionedGraph p_graph1, p_graph2;
    tbb::parallel_invoke(
        [&] { p_graph1 = partition_recursive(subgraphs[0], 2 * current_block, current_k * 2); },
        [&] { p_graph2 = partition_recursive(subgraphs[1], 2 * current_block, current_k * 2); }
    );
    ScalableVector<StaticArray<BlockID>> subgraph_partitions(2);
    subgraph_partitions[0] = p_graph1.take_raw_partition();
    subgraph_partitions[1] = p_graph2.take_raw_partition();

    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph), subgraph_partitions, 2 * current_k, _input_ctx.partition.k, mapping
    );
  }

  return p_graph;
}

PartitionedGraph RBMultilevelPartitioner::bipartition(
    const Graph &graph, const BlockID current_block, const BlockID current_k
) {
  // set k to 2 for max cluster weight computation
  PartitionContext bipart_ctx = _input_ctx.partition;
  bipart_ctx.k = 2;
  auto coarsener = factory::create_coarsener(_input_ctx, bipart_ctx);
  coarsener->initialize(&graph);

  PartitionContext p_ctx =
      partitioning::create_twoway_context(_input_ctx, current_block, current_k, graph);

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
