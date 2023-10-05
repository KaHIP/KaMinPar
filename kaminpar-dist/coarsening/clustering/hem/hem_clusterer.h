/*******************************************************************************
 * Clusterer via heavy edge matching.
 *
 * @file:   hem_clusterer.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
class HEMClusterer : public GlobalClusterer {
public:
  HEMClusterer(const Context &ctx);

  HEMClusterer(const HEMClusterer &) = delete;
  HEMClusterer &operator=(const HEMClusterer &) = delete;

  HEMClusterer(HEMClusterer &&) noexcept = default;
  HEMClusterer &operator=(HEMClusterer &&) = delete;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  void compute_local_matching(ColorID c, GlobalNodeWeight max_cluster_weight);
  void resolve_global_conflicts(ColorID c);

  bool validate_matching();

  const Context &_input_ctx;
  const HEMCoarseningContext &_ctx;

  const DistributedGraph *_graph;

  ClusterArray _matching;

  NoinitVector<std::uint8_t> _color_blacklist;
  NoinitVector<ColorID> _color_sizes;
  NoinitVector<NodeID> _color_sorted_nodes;
};
} // namespace kaminpar::dist
