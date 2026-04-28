/*******************************************************************************
 * Repair-certified cut-packet refiner.
 *
 * @file:   rccp_refiner.h
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

template <typename Graph> class RccpRefinerImpl;

class RccpRefiner : public Refiner {
  using RccpRefinerCSRImpl = RccpRefinerImpl<CSRGraph>;
  using RccpRefinerCompressedImpl = RccpRefinerImpl<CompressedGraph>;

public:
  explicit RccpRefiner(const Context &ctx);
  ~RccpRefiner() override;

  RccpRefiner(const RccpRefiner &) = delete;
  RccpRefiner &operator=(const RccpRefiner &) = delete;

  RccpRefiner(RccpRefiner &&) noexcept = default;
  RccpRefiner &operator=(RccpRefiner &&) = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::unique_ptr<RccpRefinerCSRImpl> _csr_impl;
  std::unique_ptr<RccpRefinerCompressedImpl> _compressed_impl;
};

} // namespace kaminpar::shm
