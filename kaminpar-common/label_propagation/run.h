/*******************************************************************************
 * Label propagation pass dispatch and traversal utilities.
 *
 * @file:   run.h
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-common/label_propagation/passes/growing_hash_tables.h"
#include "kaminpar-common/label_propagation/passes/single_phase.h"
#include "kaminpar-common/label_propagation/passes/two_phase.h"

namespace kaminpar::lp {

template <typename Order, typename Kernel, typename Pass>
void run_pass_iteration(Order &order, Kernel &kernel, Pass &pass) {
  auto make_local = [&] {
    return pass.local();
  };
  auto handle_node = [](auto &local, const auto node) {
    local.handle_next_node(node);
  };
  auto should_skip = [&] {
    return kernel.should_stop();
  };

  if constexpr (requires {
                  order.parallel_for_each_with_local(make_local, handle_node, should_skip);
                }) {
    order.parallel_for_each_with_local(make_local, handle_node, should_skip);
  } else if constexpr (requires { order.parallel_for_each_with_local(make_local, handle_node); }) {
    order.parallel_for_each_with_local(make_local, handle_node);
  } else {
    order.parallel_for_each([&](const auto node) {
      if (kernel.should_stop()) {
        return;
      }
      auto local = pass.local();
      local.handle_next_node(node);
    });
  }
}

template <typename Kernel, typename Callback>
decltype(auto) with_pass(Kernel &kernel, const ExecutionConfig &execution, Callback &&callback) {
  switch (execution.strategy) {
  case RatingMapStrategy::GROWING_HASH_TABLES:
    switch (kernel.config().selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC: {
      GrowingHashTablePass<Kernel, TieBreakingStrategy::GEOMETRIC> pass(kernel);
      return std::forward<Callback>(callback)(pass);
    }
    case TieBreakingStrategy::UNIFORM: {
      GrowingHashTablePass<Kernel, TieBreakingStrategy::UNIFORM> pass(kernel);
      return std::forward<Callback>(callback)(pass);
    }
    }

  case RatingMapStrategy::SINGLE_PHASE:
    switch (kernel.config().selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC: {
      SinglePhasePass<Kernel, TieBreakingStrategy::GEOMETRIC> pass(kernel);
      return std::forward<Callback>(callback)(pass);
    }
    case TieBreakingStrategy::UNIFORM: {
      SinglePhasePass<Kernel, TieBreakingStrategy::UNIFORM> pass(kernel);
      return std::forward<Callback>(callback)(pass);
    }
    }

  case RatingMapStrategy::TWO_PHASE:
    switch (kernel.config().selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC: {
      TwoPhasePass<Kernel, TieBreakingStrategy::GEOMETRIC> pass(kernel, execution);
      return std::forward<Callback>(callback)(pass);
    }
    case TieBreakingStrategy::UNIFORM: {
      TwoPhasePass<Kernel, TieBreakingStrategy::UNIFORM> pass(kernel, execution);
      return std::forward<Callback>(callback)(pass);
    }
    }
  }

  __builtin_unreachable();
}

template <typename Order, typename Kernel>
typename Kernel::Result
run_iteration(Order &order, Kernel &kernel, const ExecutionConfig &execution = {}) {
  return with_pass(kernel, execution, [&](auto &pass) {
    run_pass_iteration(order, kernel, pass);
    return pass.finish();
  });
}

} // namespace kaminpar::lp
