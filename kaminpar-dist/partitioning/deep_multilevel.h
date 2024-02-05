/*******************************************************************************
 * Deep multilevel graph partitioning with direct k-way initial partitioning.
 * @file:   deep_multilevel_partitioner.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#pragma once

#include <list>
#include <stack>

#include "kaminpar-dist/coarsening/coarsener.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/partitioning/partitioner.h"

namespace kaminpar::dist {
class DeepMultilevelPartitioner : public Partitioner {
public:
  DeepMultilevelPartitioner(const DistributedGraph &input_graph, const Context &input_ctx);

  DeepMultilevelPartitioner(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner &operator=(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner(DeepMultilevelPartitioner &&) noexcept = default;
  DeepMultilevelPartitioner &operator=(DeepMultilevelPartitioner &&) = delete;

  DistributedPartitionedGraph partition() final;

private:
  [[nodiscard]] Coarsener *get_current_coarsener();
  [[nodiscard]] const Coarsener *get_current_coarsener() const;

  const DistributedGraph &_input_graph;
  const Context &_input_ctx;

  std::list<DistributedGraph> _replicated_graphs;
  std::stack<Coarsener> _coarseners;
};
} // namespace kaminpar::dist
