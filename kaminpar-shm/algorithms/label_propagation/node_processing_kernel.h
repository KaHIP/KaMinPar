/*******************************************************************************
 * Adapter between reusable iteration orders and LP node processors.
 *
 * @file:   node_processing_kernel.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/inline.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

enum class NodeProcessingPhase {
  SINGLE,
  FIRST
};

template <typename Processor, NodeProcessingPhase kPhase> class NodeProcessingKernel {
public:
  explicit NodeProcessingKernel(Processor &processor) : _processor(processor) {}

  [[nodiscard]] bool should_stop() const {
    return _processor.should_stop();
  }

  class Local {
  public:
    Local(Processor &processor, typename Processor::LocalWorker worker)
        : _processor(processor),
          _worker(std::move(worker)) {}

    KAMINPAR_INLINE bool operator()(const NodeID u) {
      if (!_processor.should_visit(u)) {
        return false;
      }

      if constexpr (kPhase == NodeProcessingPhase::SINGLE) {
        _processor.visit(u, _worker);
      } else {
        _processor.visit_first_phase(u, _worker);
      }
      return true;
    }

    [[nodiscard]] bool should_stop(const NodeID u) {
      return _processor.should_stop(_worker, u);
    }

    void finish() {
      _processor.finish_work_unit(_worker);
    }

  private:
    Processor &_processor;
    typename Processor::LocalWorker _worker;
  };

  [[nodiscard]] Local make_local(Random &random) {
    return Local(_processor, _processor.make_local_worker(random));
  }

private:
  Processor &_processor;
};

template <typename Processor> [[nodiscard]] auto make_single_phase_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::SINGLE>(processor);
}

template <typename Processor> [[nodiscard]] auto make_first_phase_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::FIRST>(processor);
}

} // namespace kaminpar::shm::lp
