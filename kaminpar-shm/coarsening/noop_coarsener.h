/*******************************************************************************
 * Coarsener that does nothing, i.e., turns the whole partitioner single-level.
 *
 * @file:   noop_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class NoopCoarsener : public Coarsener {
public:
  void initialize(const Graph *graph) final {
    _graph = graph;
  }

  bool coarsen() final {
    return false;
  }

  [[nodiscard]] std::size_t level() const final {
    return 0;
  }

  [[nodiscard]] const Graph &current() const final {
    return *_graph;
  }

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final {
    return std::move(p_graph);
  }

  void release_allocated_memory() final {}

private:
  const Graph *_graph = nullptr;
};
} // namespace kaminpar::shm
