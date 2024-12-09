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
  using KaMinPar::compute_partition;

public:
  KaMinParNetworKit(int num_threads, const kaminpar::shm::Context &ctx);

  KaMinParNetworKit(const KaMinParNetworKit &) = delete;
  KaMinParNetworKit &operator=(const KaMinParNetworKit &) = delete;

  KaMinParNetworKit(KaMinParNetworKit &&) noexcept = default;
  KaMinParNetworKit &operator=(KaMinParNetworKit &&) noexcept = default;

  void copy_graph(const NetworKit::Graph &graph);

  NetworKit::Partition compute_partition(shm::BlockID k);
  NetworKit::Partition compute_partition(shm::BlockID k, double epsilon);
  NetworKit::Partition compute_partition(std::vector<double> max_block_weight_factors);
  NetworKit::Partition compute_partition(std::vector<shm::BlockWeight> max_block_weights);
};

} // namespace kaminpar
