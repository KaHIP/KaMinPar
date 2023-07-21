/*******************************************************************************
 * Deep multilevel graph partitioning with direct k-way initial partitioning.
 * @file:   deep_multilevel_partitioner.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#pragma once

#include <list>
#include <stack>

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/partitioning/partitioner.h"

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
  void print_coarsening_level(GlobalNodeWeight max_cluster_weight) const;
  void print_coarsening_converged() const;
  void print_coarsening_terminated(GlobalNodeID desired_num_nodes) const;
  void print_initial_partitioning_result(
      const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  ) const;

  Coarsener *get_current_coarsener();
  const Coarsener *get_current_coarsener() const;

  const DistributedGraph &_input_graph;
  const Context &_input_ctx;

  std::list<DistributedGraph> _replicated_graphs;
  std::stack<Coarsener> _coarseners;
};
} // namespace kaminpar::dist
