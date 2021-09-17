/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "kaminpar/algorithm/graph_extraction.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/initial_partitioning/initial_partitioning_facade.h"
#include "kaminpar/partitioning_scheme/helper.h"

#include <tbb/parallel_invoke.h>

namespace kaminpar::partitioning {
class ParallelSimpleRecursiveBisection {
public:
  ParallelSimpleRecursiveBisection(const Graph &input_graph, const Context &input_ctx)
      : _input_graph{input_graph},
        _input_ctx{input_ctx} {}

  PartitionedGraph partition() {
    DISABLE_TIMERS();
    PartitionedGraph p_graph = partition_recursive(_input_graph, _input_ctx.partition.k);
    ENABLE_TIMERS();
    return p_graph;
  }

  PartitionedGraph partition_recursive(const Graph &graph, const BlockID k) {
    auto p_graph = bipartition(graph, k);

    if (k > 2) {
      graph::SubgraphMemory memory{p_graph.n(), k, p_graph.m(), p_graph.graph().is_node_weighted(),
                                   p_graph.graph().is_edge_weighted()};
      const auto extraction = extract_subgraphs(p_graph, memory);

      const auto &subgraphs = extraction.subgraphs;
      const auto &mapping = extraction.node_mapping;

      PartitionedGraph p_graph1, p_graph2;
      tbb::parallel_invoke([&] { p_graph1 = partition_recursive(subgraphs[0], k / 2); },
                           [&] { p_graph2 = partition_recursive(subgraphs[1], k / 2); });
      scalable_vector<StaticArray<parallel::IntegralAtomicWrapper<BlockID>>> subgraph_partitions(2);
      subgraph_partitions[0] = p_graph1.take_partition();
      subgraph_partitions[1] = p_graph2.take_partition();

      copy_subgraph_partitions(p_graph, subgraph_partitions, k, _input_ctx.partition.k, mapping);
    }

    return p_graph;
  }

  PartitionedGraph bipartition(const Graph &graph, const BlockID final_k) {
    auto coarsener = factory::create_coarsener(graph, _input_ctx.coarsening);

    // set k to 2 for max cluster weight computation
    Context pseudo_input_ctx = _input_ctx;
    pseudo_input_ctx.partition.k = 2;

    const Graph *c_graph = &graph;

    // coarsening
    PartitionContext p_ctx = create_bipartition_context(_input_ctx.partition, graph, final_k / 2, final_k / 2);
    bool shrunk = true;
    while (shrunk && c_graph->n() > 2 * _input_ctx.coarsening.contraction_limit) {
      shrunk = helper::coarsen_once(coarsener.get(), c_graph, pseudo_input_ctx, p_ctx);
      c_graph = coarsener->coarsest_graph();
    }

    // initial bipartitioning
    PartitionedGraph p_graph = helper::bipartition(c_graph, final_k, _input_ctx, ip_m_ctx_pool);
    helper::update_partition_context(p_ctx, p_graph);

    // refine
    auto refiner = factory::create_refiner(graph, p_ctx, _input_ctx.refinement);
    auto balancer = factory::create_balancer(graph, p_ctx, _input_ctx.refinement);

    while (!coarsener->empty()) {
      helper::refine(refiner.get(), balancer.get(), p_graph, p_ctx, _input_ctx.refinement);
      p_graph = helper::uncoarsen_once(coarsener.get(), std::move(p_graph), p_ctx);
    }
    helper::refine(refiner.get(), balancer.get(), p_graph, p_ctx, _input_ctx.refinement);

    return p_graph;
  }

private:
  const Graph &_input_graph;
  const Context &_input_ctx;

  GlobalInitialPartitionerMemoryPool ip_m_ctx_pool;
};
} // namespace kaminpar::partitioning