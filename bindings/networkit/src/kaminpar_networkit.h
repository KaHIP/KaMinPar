/*******************************************************************************
 * NetworKit bindings for shared-memory KaMinPar.
 *
 * @file:   kaminpar_networkit.h
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <vector>

#include <kaminpar-shm/kaminpar.h>

#include <networkit/graph/Graph.hpp>

namespace kaminpar {

class KaMinParNetworKit : public KaMinPar {
public:
  KaMinParNetworKit();
  KaMinParNetworKit(const NetworKit::Graph &G);

  KaMinParNetworKit(const KaMinParNetworKit &) = delete;
  KaMinParNetworKit &operator=(const KaMinParNetworKit &) = delete;

  KaMinParNetworKit(KaMinParNetworKit &&) noexcept = default;
  KaMinParNetworKit &operator=(KaMinParNetworKit &&) noexcept = default;

  void copyGraph(const NetworKit::Graph &G);

  void copyCSRGraph(
      std::vector<std::uint64_t> xadj,
      std::vector<std::uint64_t> adjncy,
      std::vector<std::int32_t> adjwgt
  );

  std::vector<std::uint64_t> computePartition(shm::BlockID k);
  std::vector<std::uint64_t> computePartitionWithEpsilon(shm::BlockID k, double epsilon);
  std::vector<std::uint64_t> computePartitionWithFactors(std::vector<double> maxBlockWeightFactors);
  std::vector<std::uint64_t> computePartitionWithWeights(std::vector<shm::BlockWeight> maxBlockWeights);
};

} // namespace kaminpar
