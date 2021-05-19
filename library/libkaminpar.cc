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

#include "kaminpar/application/arguments.h"
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
  Context context{Context::create_default()};
};

struct PartitionerPrivate {
  Graph graph;
  Context context;
  NodePermutations permutations;

  NodeID n;
  NodeWeight original_total_node_weight;
  double epsilon;
};

//
// Builder to create Partitioner instance
//

PartitionerBuilder::PartitionerBuilder() { _pimpl = new PartitionerBuilderPrivate{}; }
PartitionerBuilder::~PartitionerBuilder() { delete _pimpl; }

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
  partitioner._pimpl->n = _pimpl->n;
  partitioner._pimpl->graph = Graph{std::move(_pimpl->nodes), std::move(_pimpl->edges), std::move(_pimpl->node_weights),
                                    std::move(_pimpl->edge_weights), false};
  partitioner._pimpl->context = _pimpl->context;
  partitioner._pimpl->context.setup(partitioner._pimpl->graph);
  partitioner._pimpl->epsilon = partitioner._pimpl->context.partition.epsilon;
  return partitioner;
}

Partitioner PartitionerBuilder::rearrange_and_create() {
  Partitioner partitioner;
  partitioner._pimpl->n = _pimpl->n;

  // keep track of original total node weight to assign nodes before returning the final partition
  if (_pimpl->node_weights.size() == 0) {
    partitioner._pimpl->original_total_node_weight = _pimpl->n;
  } else {
    partitioner._pimpl->original_total_node_weight = parallel::accumulate(_pimpl->node_weights);
  }

  partitioner._pimpl->context = _pimpl->context;
  partitioner._pimpl->epsilon = _pimpl->context.partition.epsilon;
  partitioner._pimpl->permutations = rearrange_and_remove_isolated_nodes(true, partitioner._pimpl->context.partition,
                                                                         _pimpl->nodes, _pimpl->edges,
                                                                         _pimpl->node_weights, _pimpl->edge_weights);
  partitioner._pimpl->graph = Graph{std::move(_pimpl->nodes), std::move(_pimpl->edges), std::move(_pimpl->node_weights),
                                    std::move(_pimpl->edge_weights), true};
  partitioner._pimpl->context.setup(partitioner._pimpl->graph);
  return partitioner;
}

//
// Partitioner interface
//

Partitioner::Partitioner() { _pimpl = new PartitionerPrivate{}; }
Partitioner::~Partitioner() { delete _pimpl; }

namespace {
bool was_rearranged(PartitionerPrivate *pimpl) { return !pimpl->permutations.new_to_old.empty(); }

std::unique_ptr<BlockID[]> finalize_partition(Graph &graph, PartitionedGraph &p_graph, PartitionerPrivate *pimpl) {
  if (!was_rearranged(pimpl)) {
    std::unique_ptr<BlockID[]> new_partition = std::make_unique<BlockID[]>(graph.n());
    std::copy(p_graph.partition().begin(), p_graph.partition().end(), new_partition.get());
    return new_partition;
  }

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
  std::unique_ptr<BlockID[]> new_partition = std::make_unique<BlockID[]>(graph.n());

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

void adapt_epsilon_after_isolated_nodes_removal(const Graph &graph, PartitionContext &p_ctx,
                                                const NodeWeight original_total_node_weight) {
  const BlockID k = p_ctx.k;
  const double old_max_block_weight = (1 + p_ctx.epsilon) * std::ceil(1.0 * original_total_node_weight / k);
  const double new_epsilon = old_max_block_weight / std::ceil(1.0 * graph.total_node_weight() / k) - 1;
  p_ctx.epsilon = new_epsilon;
}
} // namespace

Partitioner &Partitioner::set_option(const std::string &name, const std::string &value) {
  Arguments args;
  app::create_context_options(_pimpl->context, args);

  // simulate argc / argv arguments
  std::string empty = "";
  std::string name_cpy = name;
  std::string value_cpy = value;

  std::vector<char *> argv(3);
  argv[0] = &empty[0];
  argv[1] = &name_cpy[0];
  argv[2] = &value_cpy[0];

  args.parse(2, argv.data(), false);

  return *this;
}

std::unique_ptr<BlockID[]> Partitioner::partition(BlockID k) const {
  _pimpl->context.partition.k = k;
  _pimpl->context.partition.epsilon = _pimpl->epsilon;
  if (was_rearranged(_pimpl)) {
    adapt_epsilon_after_isolated_nodes_removal(_pimpl->graph, _pimpl->context.partition,
                                               _pimpl->original_total_node_weight);
  }
  PartitionedGraph p_graph = partitioning::partition(_pimpl->graph, _pimpl->context);
  return finalize_partition(_pimpl->graph, p_graph, _pimpl);
}

std::size_t Partitioner::partition_size() const { return static_cast<std::size_t>(_pimpl->n); }
} // namespace libkaminpar