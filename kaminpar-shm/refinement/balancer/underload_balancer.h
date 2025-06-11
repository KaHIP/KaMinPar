/*******************************************************************************
 * MultiQueue-based balancer for greedy minimum block weight balancing.
 *
 * @file:   underload_balancer.h
 * @author: Daniel Seemaier
 * @date:   11.06.2025
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/binary_heap.h"

namespace kaminpar::shm {

class UnderloadBalancer : public Refiner {
public:
  explicit UnderloadBalancer(const Context &ctx);

  ~UnderloadBalancer() override;

  UnderloadBalancer &operator=(const UnderloadBalancer &) = delete;
  UnderloadBalancer(const UnderloadBalancer &) = delete;

  UnderloadBalancer &operator=(UnderloadBalancer &&) = delete;
  UnderloadBalancer(UnderloadBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;

  std::vector<BinaryMaxHeap<double>> _pqs;
};

} // namespace kaminpar::shm
