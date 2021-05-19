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
#include "libkaminpar.h"

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/io.h"
#include "kaminpar/partitioning_scheme/partitioning.h"

#include <tbb/parallel_for.h>

namespace libkaminpar {
using namespace kaminpar;

struct PartitionerBuilderPrivate {
  NodeID n{kInvalidNodeID};
  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
  Context context;
};

struct PartitionerPrivate {
  Graph graph;
  Context context;
  NodePermutations permutations;

  double epsilon;
  double epsilon_adaptation;
};

//
// Builder to create Partitioner instance
//

PartitionerBuilder PartitionerBuilder::from_graph_file(const std::string &filename) {
  PartitionerBuilder builder;
  io::metis::read(filename, builder._pimpl->nodes, builder._pimpl->edges, builder._pimpl->node_weights,
                  builder._pimpl->edge_weights);
  builder._pimpl->n = builder._pimpl->nodes.size() - 1;
  return builder;
}

PartitionerBuilder PartitionerBuilder::from_adjacency_array(NodeID n, EdgeID *nodes, NodeID *edges) {
  const EdgeID m = nodes[n];

  PartitionerBuilder builder;
  builder._pimpl->n = n;
  builder._pimpl->nodes = StaticArray<EdgeID>(n + 1, nodes);
  builder._pimpl->edges = StaticArray<NodeID>(m, edges);
  return builder;
}

void PartitionerBuilder::with_node_weights(NodeWeight *node_weights) {
  _pimpl->node_weights = StaticArray<NodeWeight>(_pimpl->n, node_weights);
}

void PartitionerBuilder::with_edge_weights(EdgeWeight *edge_weights) {
  const EdgeID m = _pimpl->nodes[_pimpl->n];
  _pimpl->edge_weights = StaticArray<EdgeWeight>(m, edge_weights);
}

Partitioner PartitionerBuilder::create() {
  Partitioner partitioner;
  partitioner._pimpl->graph = Graph{std::move(_pimpl->nodes), std::move(_pimpl->edges), std::move(_pimpl->node_weights),
                                    std::move(_pimpl->edge_weights), false};
  partitioner._pimpl->context = _pimpl->context;
  partitioner._pimpl->context.setup(partitioner._pimpl->graph);
  partitioner._pimpl->epsilon = partitioner._pimpl->context.partition.epsilon;
  partitioner._pimpl->epsilon_adaptation = 1.0;
  return partitioner;
}

Partitioner PartitionerBuilder::rearrange_and_create() {
  Partitioner partitioner;
  partitioner._pimpl->context = _pimpl->context;
  partitioner._pimpl->epsilon = _pimpl->context.partition.epsilon;
  partitioner._pimpl->permutations = rearrange_and_remove_isolated_nodes(true, partitioner._pimpl->context.partition,
                                                                         _pimpl->nodes, _pimpl->edges,
                                                                         _pimpl->node_weights, _pimpl->edge_weights);
  partitioner._pimpl->epsilon_adaptation = _pimpl->context.partition.epsilon / partitioner._pimpl->epsilon;
  partitioner._pimpl->graph = Graph{std::move(_pimpl->nodes), std::move(_pimpl->edges), std::move(_pimpl->node_weights),
                                    std::move(_pimpl->edge_weights), true};
  partitioner._pimpl->context.setup(partitioner._pimpl->graph);
  return partitioner;
}

//
// Partitioner interface
//

namespace {
StaticArray<BlockID> build_partition_for_original_graph(Graph &graph, PartitionedGraph &p_graph,
                                                        PartitionerPrivate *pimpl) {
  const NodeID restricted_n = graph.n();

  const std::size_t restricted_nodes_size = graph.raw_nodes().size();
  const std::size_t restricted_node_weights_size = graph.raw_node_weights().size();
  graph.raw_nodes().unrestrict();
  graph.raw_node_weights().unrestrict();
  graph.update_total_node_weight();

  PartitionContext p_ctx = pimpl->context.partition;
  p_ctx.epsilon = pimpl->epsilon;
  p_ctx.setup(graph);

  // copy partition to tmp buffer
  StaticArray<BlockID> new_partition(graph.n());

  // rearrange partition for original graph
  tbb::parallel_for(static_cast<NodeID>(0), restricted_n, [&](const NodeID u) {
    const NodeID u_prime = pimpl->permutations.old_to_new[u];
    if (u_prime < restricted_n) { new_partition[u] = p_graph.block(u_prime); }
  });

  const BlockID k = p_graph.k();
  auto block_weights = p_graph.take_block_weights();
  BlockID b = 0;

  // place isolated nodes into blocks
  for (NodeID u_prime = restricted_n; u_prime < graph.n(); ++u_prime) {
    const NodeID u = pimpl->permutations.new_to_old[u_prime];
    while (b + 1 < k && block_weights[b] + graph.node_weight(u_prime) > p_ctx.max_block_weight(b)) { ++b; }
    new_partition[u] = b;
    block_weights[b] += graph.node_weight(u);
  }

  graph.raw_nodes().restrict(restricted_nodes_size);
  graph.raw_node_weights().restrict(restricted_node_weights_size);
  graph.update_total_node_weight();

  return new_partition;
}

bool was_rearranged(PartitionerPrivate *pimpl) { return !pimpl->permutations.new_to_old.empty(); }
} // namespace

void Partitioner::set_option(const std::string &name, const std::string &value) {
  // special treatment for epsilon to catch the case where we modified epsilon during preprocessing
  if (name == "epsilon") {
    _pimpl->epsilon = std::strtod(value.c_str(), nullptr);
    _pimpl->context.partition.epsilon = _pimpl->epsilon * _pimpl->epsilon_adaptation;
  } else {
    // TODO
  }
}

std::unique_ptr<BlockID> Partitioner::partition(BlockID k) {
  _pimpl->context.partition.k = k;
  PartitionedGraph p_graph = partitioning::partition(_pimpl->graph, _pimpl->context);

  auto partition = (was_rearranged(_pimpl.get()))
                       ? build_partition_for_original_graph(_pimpl->graph, p_graph, _pimpl.get())
                       : p_graph.take_partition();

  return std::unique_ptr<BlockID>(partition.free().release());
}
} // namespace libkaminpar