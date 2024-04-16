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
    : _csr_impl(std::make_unique<LabelPropagationRefinerImpl<CSRGraph>>(ctx)),
      _compact_csr_impl(std::make_unique<LabelPropagationRefinerImpl<CompactCSRGraph>>(ctx)),
      _compressed_impl(std::make_unique<LabelPropagationRefinerImpl<CompressedGraph>>(ctx)) {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

void LabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  const Graph &graph = p_graph.graph();

  if (auto *csr_graph = dynamic_cast<const CSRGraph *>(graph.underlying_graph());
      csr_graph != nullptr) {
    _csr_impl->initialize(csr_graph);
  } else if (auto *compact_csr_graph =
                 dynamic_cast<const CompactCSRGraph *>(graph.underlying_graph());
             compact_csr_graph != nullptr) {
    _compact_csr_impl->initialize(compact_csr_graph);
  } else if (auto *compressed_graph =
                 dynamic_cast<const CompressedGraph *>(graph.underlying_graph());
             compressed_graph != nullptr) {
    _compressed_impl->initialize(compressed_graph);
  }
}

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  const auto specific_refine = [&](auto &impl) {
    if (_freed) {
      _freed = false;
      impl.allocate();
    } else {
      impl.setup(std::move(_structs));
    }

    const bool found_improvement = impl.refine(p_graph, p_ctx);

    _structs = impl.release();
    return found_improvement;
  };

  SCOPED_TIMER("Label Propagation");
  const Graph &graph = p_graph.graph();

  if (auto *csr_graph = dynamic_cast<const CSRGraph *>(graph.underlying_graph());
      csr_graph != nullptr) {
    return specific_refine(*_csr_impl);
  }

  if (auto *compact_csr_graph = dynamic_cast<const CompactCSRGraph *>(graph.underlying_graph());
      compact_csr_graph != nullptr) {
    return specific_refine(*_compact_csr_impl);
  }

  if (auto *compressed_graph = dynamic_cast<const CompressedGraph *>(graph.underlying_graph());
      compressed_graph != nullptr) {
    return specific_refine(*_compressed_impl);
  }

  __builtin_unreachable();
}
} // namespace kaminpar::shm
