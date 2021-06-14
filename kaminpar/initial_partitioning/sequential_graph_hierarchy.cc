/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
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

#include "sequential_graph_hierarchy.h"

namespace kaminpar::ip {
SequentialGraphHierarchy::SequentialGraphHierarchy(const Graph *finest_graph) : _finest_graph(finest_graph) {}

void SequentialGraphHierarchy::take_coarse_graph(Graph &&c_graph, std::vector<NodeID> &&c_mapping) {
  ASSERT(coarsest_graph().n() == c_mapping.size());
  _coarse_mappings.push_back(std::move(c_mapping));
  _coarse_graphs.push_back(std::move(c_graph));
}

[[nodiscard]] const Graph &SequentialGraphHierarchy::coarsest_graph() const {
  return _coarse_graphs.empty() ? *_finest_graph : _coarse_graphs.back();
}

PartitionedGraph SequentialGraphHierarchy::pop_and_project(PartitionedGraph &&coarse_p_graph) {
  ASSERT(!_coarse_graphs.empty());
  ASSERT(&_coarse_graphs.back() == &coarse_p_graph.graph());

  // goal: project partition of p_graph == c_graph onto new_c_graph
  std::vector<NodeID> c_mapping{std::move(_coarse_mappings.back())};
  _coarse_mappings.pop_back();

  const Graph &graph{get_second_coarsest_graph()};
  ASSERT(graph.n() == c_mapping.size());

  StaticArray<BlockID> partition{graph.n()};
  for (const NodeID u : graph.nodes()) { partition[u] = coarse_p_graph.block(c_mapping[u]); }

  // this destroys underlying Graph wrapped in p_graph
  _coarse_graphs.pop_back();

  PartitionedGraph p_graph{tag::seq, graph, coarse_p_graph.k(), std::move(partition),
                           std::move(coarse_p_graph.take_final_k())};
#ifdef KAMIPAR_GRAPH_NAMES
  p_graph.set_block_names(coarse_p_graph.block_names());
#endif // KAMIPAR_GRAPH_NAMES
  return p_graph;
}

const Graph &SequentialGraphHierarchy::get_second_coarsest_graph() const {
  ASSERT(!_coarse_graphs.empty());
  return (_coarse_graphs.size() > 1) ? _coarse_graphs[_coarse_graphs.size() - 2] : *_finest_graph;
}
} // namespace kaminpar::ip
