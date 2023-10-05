/*******************************************************************************
 * Adaptive stopping policies for the parallel k-way FM refinement algorithm.
 *
 * @file:   stopping_policies.h
 * @author: Daniel Seemaier
 * @date:   12.04.2023
 ******************************************************************************/
#pragma once

#include <cmath>

#include "kaminpar-shm/definitions.h"

namespace kaminpar::shm {
struct AdaptiveStoppingPolicy {
  AdaptiveStoppingPolicy(const double alpha) : _factor(alpha / 2.0 - 0.25) {}

  void init(const NodeID n) {
    _beta = std::log(n);
  }

  [[nodiscard]] bool should_stop() const {
    return (_num_steps > _beta) &&
           ((_Mk == 0) || (_num_steps >= (_variance / (_Mk * _Mk)) * _factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const EdgeWeight gain) {
    ++_num_steps;

    // See Knuth TAOCP vol 2, 3rd edition, page 232 or
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    if (_num_steps == 1) {
      _MkMinus1 = static_cast<double>(gain);
      _Mk = _MkMinus1;
      _SkMinus1 = 0.0;
    } else {
      _Mk = _MkMinus1 + (gain - _MkMinus1) / _num_steps;
      _Sk = _SkMinus1 + (gain - _MkMinus1) * (gain - _Mk);
      _variance = _Sk / (_num_steps - 1.0);

      // Prepare for next iteration
      _MkMinus1 = _Mk;
      _SkMinus1 = _Sk;
    }
  }

private:
  double _factor;

  double _beta = 0.0;
  std::size_t _num_steps = 0;
  double _variance = 0.0;
  double _Mk = 0.0;
  double _MkMinus1 = 0.0;
  double _Sk = 0.0;
  double _SkMinus1 = 0.0;
};
} // namespace kaminpar::shm
