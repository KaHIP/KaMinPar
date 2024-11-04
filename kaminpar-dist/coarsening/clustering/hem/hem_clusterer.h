/*******************************************************************************
 * Clusterer via heavy edge matching.
 *
 * @file:   hem_clusterer.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {

class HEMClusterer : public Clusterer {
public:
  explicit HEMClusterer(const Context &ctx);

  HEMClusterer(const HEMClusterer &) = delete;
  HEMClusterer &operator=(const HEMClusterer &) = delete;

  HEMClusterer(HEMClusterer &&) noexcept = default;
  HEMClusterer &operator=(HEMClusterer &&) = delete;

  ~HEMClusterer() override;

  void set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) final;

  void cluster(StaticArray<GlobalNodeID> &matching, const DistributedGraph &graph) final;

private:
  std::unique_ptr<class HEMClustererImplWrapper> _impl_wrapper;
};

} // namespace kaminpar::dist
