/*******************************************************************************
 * Partitioning scheme that uses toplevel multilevel recursvie bipartitioning.
 *
 * @file:   rb_multilevel.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/initial_partitioning/initial_partitioning_facade.h"
#include "kaminpar-shm/partition_utils.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partitioner.h"

namespace kaminpar::shm {
class RBMultilevelPartitioner : public Partitioner {
public:
  RBMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx)
      : _input_graph(input_graph),
        _input_ctx(input_ctx) {}

  PartitionedGraph partition() final {
    DISABLE_TIMERS();
    PartitionedGraph p_graph = partition_recursive(_input_graph, _input_ctx.partition.k);
    ENABLE_TIMERS();
    return p_graph;
  }

  PartitionedGraph partition_recursive(const Graph &graph, const BlockID k) {
    auto p_graph = bipartition(graph, k);

    if (k > 2) {
      graph::SubgraphMemory memory;
      memory.resize(
          p_graph.n(),
          k,
          p_graph.m(),
          p_graph.graph().node_weighted(),
          p_graph.graph().edge_weighted()
      );

      const auto extraction = extract_subgraphs(p_graph, _input_ctx.partition.k, memory);

      const auto &subgraphs = extraction.subgraphs;
      const auto &mapping = extraction.node_mapping;

      PartitionedGraph p_graph1, p_graph2;
      tbb::parallel_invoke(
          [&] { p_graph1 = partition_recursive(subgraphs[0], k / 2); },
          [&] { p_graph2 = partition_recursive(subgraphs[1], k / 2); }
      );
      scalable_vector<StaticArray<BlockID>> subgraph_partitions(2);
      subgraph_partitions[0] = p_graph1.take_raw_partition();
      subgraph_partitions[1] = p_graph2.take_raw_partition();

      p_graph = graph::copy_subgraph_partitions(
          std::move(p_graph), subgraph_partitions, k, _input_ctx.partition.k, mapping
      );
    }

    return p_graph;
  }

  PartitionedGraph bipartition(const Graph &graph, const BlockID final_k) {
    using namespace partitioning;

    auto coarsener = factory::create_coarsener(graph, _input_ctx.coarsening);

    // set k to 2 for max cluster weight computation
    Context pseudo_input_ctx = _input_ctx;
    pseudo_input_ctx.partition.k = 2;

    const Graph *c_graph = &graph;

    // coarsening
    PartitionContext p_ctx =
        create_bipartition_context(graph, final_k / 2, final_k / 2, _input_ctx.partition);
    bool shrunk = true;
    while (shrunk && c_graph->n() > 2 * _input_ctx.coarsening.contraction_limit) {
      shrunk = helper::coarsen_once(coarsener.get(), c_graph, pseudo_input_ctx, p_ctx);
      c_graph = coarsener->coarsest_graph();
    }

    // initial bipartitioning
    PartitionedGraph p_graph = helper::bipartition(c_graph, final_k, _input_ctx, ip_m_ctx_pool);
    helper::update_partition_context(p_ctx, p_graph, _input_ctx.partition.k);

    // refine
    auto refiner = factory::create_refiner(_input_ctx);

    while (!coarsener->empty()) {
      helper::refine(refiner.get(), p_graph, p_ctx);
      p_graph =
          helper::uncoarsen_once(coarsener.get(), std::move(p_graph), p_ctx, _input_ctx.partition);
    }
    helper::refine(refiner.get(), p_graph, p_ctx);

    return p_graph;
  }

private:
  const Graph &_input_graph;
  const Context &_input_ctx;

  partitioning::GlobalInitialPartitionerMemoryPool ip_m_ctx_pool;
};
} // namespace kaminpar::shm
