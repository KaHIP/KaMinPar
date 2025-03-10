/*******************************************************************************
 * NetworKit bindings for shared-memory KaMinPar.
 *
 * @file:   kaminpar_networkit.h
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#pragma once

#include <kaminpar-shm/kaminpar.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/structures/Partition.hpp>

namespace kaminpar {

class KaMinParNetworKit : public KaMinPar {
public:
  KaMinParNetworKit(const NetworKit::Graph &G);

  KaMinParNetworKit(const KaMinParNetworKit &) = delete;
  KaMinParNetworKit &operator=(const KaMinParNetworKit &) = delete;

  KaMinParNetworKit(KaMinParNetworKit &&) noexcept = default;
  KaMinParNetworKit &operator=(KaMinParNetworKit &&) noexcept = default;

  void copyGraph(const NetworKit::Graph &G);

  NetworKit::Partition computePartition(shm::BlockID k);
  NetworKit::Partition computePartitionWithEpsilon(shm::BlockID k, double epsilon);
  NetworKit::Partition computePartitionWithFactors(std::vector<double> maxBlockWeightFactors);
  NetworKit::Partition computePartitionWithWeights(std::vector<shm::BlockWeight> maxBlockWeights);
};

} // namespace kaminpar
