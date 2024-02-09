/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

namespace kaminpar::shm {
LabelPropagationRefiner::LabelPropagationRefiner(const Context &ctx)
    : _csr_impl{std::make_unique<LabelPropagationRefinerImpl<CSRGraph>>(ctx)},
      _compact_csr_impl{std::make_unique<LabelPropagationRefinerImpl<CompactCSRGraph>>(ctx)},
      _compressed_impl{std::make_unique<LabelPropagationRefinerImpl<CompressedGraph>>(ctx)} {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

void LabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  const Graph &graph = p_graph.graph();

  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    _csr_impl->initialize(csr_graph);
  } else if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
             compact_csr_graph != nullptr) {
    _compact_csr_impl->initialize(compact_csr_graph);
  } else if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
             compressed_graph != nullptr) {
    _compressed_impl->initialize(compressed_graph);
  }
}

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  const auto refine_specific_impl = [&](auto *impl) {
    if (!_allocated) {
      _allocated = true;
      impl->allocate();
    } else {
      impl->setup(std::move(_structs));
    }

    const bool found_improvement = impl->refine(p_graph, p_ctx);

    _structs = impl->release();
    return found_improvement;
  };

  const Graph &graph = p_graph.graph();

  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    return refine_specific_impl(_csr_impl.get());
  }

  if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
      compact_csr_graph != nullptr) {
    return refine_specific_impl(_compact_csr_impl.get());
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
      compressed_graph != nullptr) {
    return refine_specific_impl(_compressed_impl.get());
  }

  __builtin_unreachable();
}
} // namespace kaminpar::shm
