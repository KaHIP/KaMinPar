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

#pragma once

#include "datastructure/graph.h"

namespace kaminpar::ip {
class SequentialGraphHierarchy {
public:
  explicit SequentialGraphHierarchy(const Graph *finest_graph);

  SequentialGraphHierarchy(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy &operator=(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy(SequentialGraphHierarchy &&) noexcept = default;
  SequentialGraphHierarchy &operator=(SequentialGraphHierarchy &&) noexcept = default;

  void take_coarse_graph(Graph &&c_graph, std::vector<NodeID> &&c_mapping);

  [[nodiscard]] const Graph &coarsest_graph() const;

  PartitionedGraph pop_and_project(PartitionedGraph &&coarse_p_graph);

  [[nodiscard]] inline std::size_t size() const { return _coarse_graphs.size(); }
  [[nodiscard]] inline bool empty() const { return _coarse_graphs.empty(); }
  [[nodiscard]] inline const auto &coarse_mappings() const { return _coarse_mappings; }
  [[nodiscard]] inline const auto &coarse_graphs() const { return _coarse_graphs; }

private:
  const Graph &get_second_coarsest_graph() const;

  const Graph *_finest_graph;
  std::vector<std::vector<NodeID>> _coarse_mappings;
  std::vector<Graph> _coarse_graphs;
};
} // namespace kaminpar::ip