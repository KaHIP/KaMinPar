/*******************************************************************************
 * Interface for initial refinement algorithms.
 *
 * @file:   initial_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class InitialRefiner {
public:
  virtual ~InitialRefiner() = default;

  virtual void init(const CSRGraph &graph) = 0;
  virtual bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) = 0;
};

class InitialMultiRefiner : public InitialRefiner {
public:
  InitialMultiRefiner(
      std::unordered_map<InitialRefinementAlgorithm, std::unique_ptr<InitialRefiner>> refiners,
      std::vector<InitialRefinementAlgorithm> order
  );

  InitialMultiRefiner(const InitialMultiRefiner &) = delete;
  InitialMultiRefiner &operator=(const InitialMultiRefiner &) = delete;

  InitialMultiRefiner(InitialMultiRefiner &&) = default;
  InitialMultiRefiner &operator=(InitialMultiRefiner &&) = default;

  void init(const CSRGraph &graph) override;
  bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  std::unordered_map<InitialRefinementAlgorithm, std::unique_ptr<InitialRefiner>> _refiners;
  std::vector<InitialRefinementAlgorithm> _order;

  const CSRGraph *_graph;
};

std::unique_ptr<InitialRefiner> create_initial_refiner(const InitialRefinementContext &r_ctx);

} // namespace kaminpar::shm
