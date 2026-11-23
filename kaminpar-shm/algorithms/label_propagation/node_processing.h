/*******************************************************************************
 * Shared execution support for label-propagation node processors.
 *
 * @file:   node_processing.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/inline.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

struct RoundStats {
  NodeID moved = 0;
};

class RoundStatistics {
public:
  [[nodiscard]] RoundStats &local() {
    return _locals.local();
  }

  [[nodiscard]] RoundStats totals() {
    return _locals.combine([](const RoundStats lhs, const RoundStats rhs) {
      return RoundStats{lhs.moved + rhs.moved};
    });
  }

  void clear() {
    _locals.clear();
  }

private:
  tbb::enumerable_thread_specific<RoundStats> _locals;
};

enum class NodeProcessingPhase {
  SINGLE,
  FIRST,
  SINGLE_FINAL,
  FIRST_FINAL,
  ALL_FIRST_ITER,
  ALL_FIRST_AND_LAST,
  LAST_ITER
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
      if constexpr (kPhase == NodeProcessingPhase::ALL_FIRST_ITER) {
        _processor.visit_first_iter(u, _worker);
        return true;
      } else if constexpr (kPhase == NodeProcessingPhase::ALL_FIRST_AND_LAST) {
        _processor.visit_first_and_last_iter(u, _worker);
        return true;
      } else {
        if (!_processor.should_visit(u)) {
          return false;
        }

        if constexpr (kPhase == NodeProcessingPhase::SINGLE) {
          _processor.visit(u, _worker);
        } else if constexpr (kPhase == NodeProcessingPhase::FIRST) {
          _processor.visit_first_phase(u, _worker);
        } else if constexpr (kPhase == NodeProcessingPhase::SINGLE_FINAL) {
          _processor.visit_final(u, _worker);
        } else if constexpr (kPhase == NodeProcessingPhase::FIRST_FINAL) {
          _processor.visit_first_phase_final(u, _worker);
        } else {
          _processor.visit_last_iter(u, _worker);
        }
        return true;
      }
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

template <typename Processor>
[[nodiscard]] auto make_single_phase_final_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::SINGLE_FINAL>(processor);
}

template <typename Processor>
[[nodiscard]] auto make_first_phase_final_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::FIRST_FINAL>(processor);
}

template <typename Processor>
[[nodiscard]] auto make_first_iter_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::ALL_FIRST_ITER>(processor);
}

template <typename Processor>
[[nodiscard]] auto make_first_and_last_iter_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::ALL_FIRST_AND_LAST>(processor);
}

template <typename Processor>
[[nodiscard]] auto make_last_iter_kernel(Processor &processor) {
  return NodeProcessingKernel<Processor, NodeProcessingPhase::LAST_ITER>(processor);
}

} // namespace kaminpar::shm::lp
