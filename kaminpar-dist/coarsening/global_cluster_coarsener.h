/*******************************************************************************
 * Graph coarsener based on global clusterings.
 *
 * @file:   global_cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/coarsening/coarsener.h"
#include "kaminpar-dist/coarsening/contraction.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
class GlobalClusterCoarsener : public Coarsener {
public:
  GlobalClusterCoarsener(const Context &input_ctx);

  void initialize(const DistributedGraph *graph) final;

  bool coarsen() final;

  [[nodiscard]] virtual std::size_t level() const final;

  [[nodiscard]] virtual const DistributedGraph &current() const final;

  DistributedPartitionedGraph uncoarsen(DistributedPartitionedGraph &&p_graph) final;

private:
  bool has_converged(const DistributedGraph &before, const DistributedGraph &after) const;
  GlobalNodeWeight max_cluster_weight() const;

  const Context &_input_ctx;

  const DistributedGraph *_input_graph = nullptr;

  std::unique_ptr<Clusterer> _clusterer;

  std::vector<std::unique_ptr<CoarseGraph>> _graph_hierarchy;
};
} // namespace kaminpar::dist

