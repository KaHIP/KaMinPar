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
    if constexpr (requires { pass.buffered_local(); }) {
      return pass.buffered_local();
    } else {
      return pass.local();
    }
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

template <
    template <typename, ActiveSetStrategy, TieBreakingStrategy> typename Pass,
    ActiveSetStrategy ActiveSet,
    typename Kernel,
    typename Callback,
    typename... Args>
decltype(auto) with_tie_breaking_pass(Kernel &kernel, Callback &&callback, Args &&...args) {
  switch (kernel.config().selection.tie_breaking_strategy) {
  case TieBreakingStrategy::GEOMETRIC: {
    Pass<Kernel, ActiveSet, TieBreakingStrategy::GEOMETRIC> pass(
        kernel, std::forward<Args>(args)...
    );
    return std::forward<Callback>(callback)(pass);
  }
  case TieBreakingStrategy::UNIFORM: {
    Pass<Kernel, ActiveSet, TieBreakingStrategy::UNIFORM> pass(kernel, std::forward<Args>(args)...);
    return std::forward<Callback>(callback)(pass);
  }
  }
  __builtin_unreachable();
}

template <
    template <typename, ActiveSetStrategy, TieBreakingStrategy> typename Pass,
    typename Kernel,
    typename Callback,
    typename... Args>
decltype(auto) with_active_set_pass(Kernel &kernel, Callback &&callback, Args &&...args) {
  switch (kernel.config().active_set.strategy) {
  case ActiveSetStrategy::NONE:
    return with_tie_breaking_pass<Pass, ActiveSetStrategy::NONE>(
        kernel, std::forward<Callback>(callback), std::forward<Args>(args)...
    );
  case ActiveSetStrategy::GLOBAL:
    return with_tie_breaking_pass<Pass, ActiveSetStrategy::GLOBAL>(
        kernel, std::forward<Callback>(callback), std::forward<Args>(args)...
    );
  case ActiveSetStrategy::LOCAL:
    return with_tie_breaking_pass<Pass, ActiveSetStrategy::LOCAL>(
        kernel, std::forward<Callback>(callback), std::forward<Args>(args)...
    );
  }
  __builtin_unreachable();
}

template <typename Kernel, typename Callback>
decltype(auto) with_pass(Kernel &kernel, const ExecutionConfig &execution, Callback &&callback) {
  switch (execution.strategy) {
  case RatingMapStrategy::GROWING_HASH_TABLES:
    return with_active_set_pass<GrowingHashTablePass>(kernel, std::forward<Callback>(callback));
  case RatingMapStrategy::SINGLE_PHASE:
    return with_active_set_pass<SinglePhasePass>(kernel, std::forward<Callback>(callback));
  case RatingMapStrategy::TWO_PHASE:
    return with_active_set_pass<TwoPhasePass>(kernel, std::forward<Callback>(callback), execution);
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
