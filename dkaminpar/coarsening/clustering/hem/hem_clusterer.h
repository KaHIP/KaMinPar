/*******************************************************************************
 * Clusterer via heavy edge matching.
 *
 * @file:   hem_clusterer.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist {
class HEMClustering : public GlobalClusterer {
public:
  HEMClustering(const Context &ctx);

  HEMClustering(const HEMClustering &) = delete;
  HEMClustering &operator=(const HEMClustering &) = delete;

  HEMClustering(HEMClustering &&) noexcept = default;
  HEMClustering &operator=(HEMClustering &&) = delete;

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
