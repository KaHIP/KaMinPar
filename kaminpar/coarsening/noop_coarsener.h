/*******************************************************************************
 * @file:   noop_coarsener.h
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Coarsener that converges immediately.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/i_coarsener.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar {
class NoopCoarsener : public ICoarsener {
public:
  void initialize(const Graph *graph) final { _graph = graph; }

  std::pair<const Graph *, bool> compute_coarse_graph(const NodeWeight /* max_cluster_weight */,
                                                      const NodeID /* to_size */) final {
    return {coarsest_graph(), false};
  }

  [[nodiscard]] std::size_t size() const final { return 0; }

  [[nodiscard]] const Graph *coarsest_graph() const final { return _graph; }

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final { return std::move(p_graph); }

private:
  const Graph *_graph{nullptr};
};
} // namespace kaminpar
