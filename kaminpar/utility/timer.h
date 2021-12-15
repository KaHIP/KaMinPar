/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "definitions.h"
#include "parallel.h"

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#define TIMER_EXTRACT_SUBGRAPHS "Subgraph extraction"
#define TIMER_COARSENING "Coarsening"
#define TIMER_UNCOARSENING "Uncoarsening"
#define TIMER_PARTITIONING "Partitioning"
#define TIMER_INITIAL_PARTITIONING "Initial partitioning"
#define TIMER_INITIAL_PARTITIONING_SCHEME "Initial partitioning scheme"
#define TIMER_CONTRACT_GRAPH "Contraction"
#define TIMER_UNCONTRACT "Uncontraction"
#define TIMER_IO "IO"
#define TIMER_ALLOCATION "Allocation"
#define TIMER_LABEL_PROPAGATION "Label propagation"
#define TIMER_REFINEMENT "Refinement"
#define TIMER_STATISTICS "Statistics"
#define TIMER_FLAT_RECURSIVE_BISECTION "Flat recursive bisection"
#define TIMER_BIPARTITIONER "Bipartitioner"
#define TIMER_COPY_SUBGRAPH_PARTITIONS "Copy subgraph partitions"
#define TIMER_BALANCER "Balancer"

#ifdef KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER_IMPL2(x, line, func)                                                                              \
  auto __SCOPED_TIMER__##line = (kaminpar::Timer::global().start_scoped_timer<func>(x))
#define SCOPED_TIMER_IMPL1(x, line, func) SCOPED_TIMER_IMPL2(x, line, func)
#define SCOPED_TIMER(x) SCOPED_TIMER_IMPL1(x, __LINE__, timer::TimerType::GLOBAL)
#define SCOPED_LOCAL_TIMER(x) SCOPED_TIMER_IMPL1(x, __LINE__, timer::TimerType::LOCAL)

#define START_TIMER(x) (kaminpar::Timer::global().start_global_timer(x))
#define STOP_TIMER() (kaminpar::Timer::global().stop_global_timer())

#define START_LOCAL_TIMER(x) (kaminpar::Timer::global().start_local_timer(x))
#define STOP_LOCAL_TIMER() (kaminpar::Timer::global().stop_local_timer())

#define ENABLE_TIMERS() (kaminpar::Timer::global().enable())
#define DISABLE_TIMERS() (kaminpar::Timer::global().disable())
#else // KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER(x)
#define START_TIMER(x)
#define STOP_TIMER()

#define ENABLE_TIMERS()
#define DISABLE_TIMERS()
#endif // KAMINPAR_ENABLE_TIMERS

// must be followed by a lambda body that may or may not return some value
#define TIMED_SCOPE(x) kaminpar::timer::TimedScope<timer::TimerType::GLOBAL>{(x)} + [&]
#define LOCAL_TIMED_SCOPE(x) kaminpar::timer::TimedScope<timer::TimerType::LOCAL>{(x)} + [&]

// macros for simple, non-hierarchical timers
// these lower performance during IP due to excessive synchronization -- redo with thread locals or remove
//#define SIMPLE_TIMER_START() (timer::now())
#define SIMPLE_TIMER_START() (0)
//#define SIMPLE_TIMER_STOP(name, start_tp) (timer::FlatTimer::global().add_timing((name), timer::now() - (start_tp)))
#define SIMPLE_TIMER_STOP(name, start_tp) ((void) start_tp)

namespace kaminpar {
namespace timer {
inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

enum class TimerType { LOCAL, GLOBAL };

class FlatTimer {
public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using Duration = std::chrono::high_resolution_clock::duration;

  static FlatTimer &global();

  void add_timing(const std::string &name, const Duration duration) {
    std::lock_guard<std::mutex> lg{_mutex};
    _timings[name] += duration;
  }

  void print(std::ostream &out) {
    for (const auto &[name, duration] : _timings) {
      out << "+-- " << name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0
          << "\n";
    }
  }

private:
  std::unordered_map<std::string, Duration> _timings{};
  std::mutex _mutex{};
};

template<TimerType type>
class ScopedTimer;
} // namespace timer

class Timer {
  static constexpr bool kDebug = false;
  static constexpr std::size_t kSpaceBetweenTimeAndRestarts = 1;

public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using Duration = std::chrono::high_resolution_clock::duration;

private:
  struct TimerTreeNode {
    std::size_t restarts{0};
    Duration elapsed{};
    TimePoint start{};
    TimerTreeNode *parent{nullptr};
    std::map<std::string, std::unique_ptr<TimerTreeNode>> children{};
    std::map<std::string, std::unique_ptr<TimerTreeNode>> local_children{};
    std::string name;

    // if we already have a subtree named `name`, merge it with the given subtree
    // otherwise, append the given subtree to this node
    void merge_append_local_subtree(const std::string &local_name, std::unique_ptr<TimerTreeNode> &&subtree) {
      if (local_children.find(local_name) != local_children.end()) {
        auto &owned_subtree = local_children[local_name];
        owned_subtree->restarts += subtree->restarts;
        owned_subtree->elapsed += subtree->elapsed;
        for (const auto &[child_name, child_subtree] : subtree->local_children) {
          owned_subtree->merge_append_local_subtree(child_name, std::move(subtree->local_children[child_name]));
        }
      } else {
        subtree->parent = this;
        local_children[local_name] = std::move(subtree);
      }
    }
  };

  struct TimerTree {
    TimerTreeNode root{};
    TimerTreeNode *current{&root};
  };

public:
  static Timer &global();

  Timer(const std::string &name);

  void start_global_timer(const std::string &name) {
    if (_disabled) { return; }

    std::lock_guard<std::mutex> lg{_global_lock};
    start_timer(_global_root, name,
                [](auto *node) -> std::map<std::string, std::unique_ptr<TimerTreeNode>> & { return node->children; });
  }

  void stop_global_timer() {
    if (_disabled) { return; }

    std::lock_guard<std::mutex> lg{_global_lock};
    stop_timer(_global_root);
  }

  std::string DBG_build_stack(const std::string &current = "") {
    TimerTreeNode *c = _local_root.local().current;
    std::stringstream ss;
    ss << current;
    while (c != nullptr) {
      ss << " -> " << c->name;
      c = c->parent;
    }
    return ss.str();
  }

  void start_local_timer(const std::string &name) {
    if (_disabled || _local_timers_disabled) { return; }

    DBG << "Thread " << sched_getcpu() << " started: " << DBG_build_stack(name);
    start_timer(_local_root.local(), name, [](auto *node) -> std::map<std::string, std::unique_ptr<TimerTreeNode>> & {
      return node->local_children;
    });
  }

  void stop_local_timer() {
    if (_disabled || _local_timers_disabled) { return; }

    TimerTree &tree = _local_root.local();
    DBG << "Thread " << sched_getcpu() << " stopped " << DBG_build_stack();
    stop_timer(tree);

    // stopped all local timers -> merge/append to global timer
    if (tree.current == &tree.root) {
      DBG << "Thread " << sched_getcpu() << " stopped _ALL_ timers -> merge with root";

      ASSERT(tree.root.local_children.size() == 1); // should have exactly one child -- the one we just stopped
      const std::string &name = tree.root.local_children.begin()->first;
      std::unique_ptr<TimerTreeNode> &subtree = tree.root.local_children.begin()->second;

      {
        std::lock_guard<std::mutex> lg{_global_lock};
        _global_root.current->merge_append_local_subtree(name, std::move(subtree));
      }

      tree.root.local_children.clear();
    }
  }

  template<timer::TimerType type>
  timer::ScopedTimer<type> start_scoped_timer(const std::string &name) {
    switch (type) {
      case timer::TimerType::GLOBAL: start_global_timer(name); break;
      case timer::TimerType::LOCAL: start_local_timer(name); break;
    }
    return timer::ScopedTimer<type>{this};
  }

  void print_machine_readable(std::ostream &out);

  void print_human_readable(std::ostream &out);

  void disable() { _disabled = true; }
  void enable() { _disabled = false; }
  void disable_local() { _local_timers_disabled = true; }
  void enable_local() { _local_timers_disabled = false; }

private:
  template<typename MapSelector>
  void start_timer(TimerTree &tree, const std::string &name, MapSelector &&selector) {
    if (selector(tree.current).find(name) == selector(tree.current).end()) {
      std::unique_ptr<TimerTreeNode> node{std::make_unique<TimerTreeNode>()};
      node->parent = tree.current;
      node->name = name;
      selector(tree.current)[name] = std::move(node);
    }
    auto *node = selector(tree.current)[name].get();
    tree.current = node;
    ++node->restarts;
    node->start = timer::now();
  }

  void stop_timer(TimerTree &tree) {
    const TimePoint end = timer::now();
    auto *node = tree.current;
    node->elapsed += end - node->start;
    tree.current = node->parent;
  }

  static void print_padded_timing(std::ostream &out, std::size_t start_column, std::size_t time_column,
                                  std::size_t time_space, const TimerTreeNode *of_process);

  void print_subprocess_children_hr(std::ostream &out, const std::string &base_prefix, const TimerTreeNode *process,
                                    std::size_t time_column, std::size_t time_space) const;

  [[nodiscard]] std::size_t compute_time_output_column(std::size_t prefix_length, const std::string &name,
                                                       const TimerTreeNode *process) const;

  [[nodiscard]] std::size_t compute_max_time_space(const TimerTreeNode *process) const;

  void print_subprocess_mr(std::ostream &out, const std::string &prefix, const std::string &name,
                           const TimerTreeNode *subtree, const bool is_local);

  std::string _name;
  tbb::enumerable_thread_specific<TimerTree> _local_root{};
  TimerTree _global_root{};
  std::mutex _global_lock;
  bool _disabled;
  bool _local_timers_disabled;
};

namespace timer {

template<TimerType type>
class ScopedTimer {
public:
  explicit ScopedTimer(Timer *timer) : _timer{timer} {}
  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
  ScopedTimer(ScopedTimer &&other) noexcept : _timer{other._timer} { other._timer = nullptr; };
  ScopedTimer &operator=(ScopedTimer &&other) noexcept { return (std::swap(_timer, other._timer), *this); };
  ~ScopedTimer() {
    if (_timer != nullptr) {
      switch (type) {
        case TimerType::GLOBAL: _timer->stop_global_timer(); break;
        case TimerType::LOCAL: _timer->stop_local_timer(); break;
      }
    }
  }

private:
  Timer *_timer{nullptr};
};

template<timer::TimerType type>
class TimedScope {
public:
  explicit TimedScope(const std::string &timer_name) : _timer_name{timer_name} {}

  template<typename F>
  decltype(auto) operator+(F &&f) {
    const auto scope = Timer::global().start_scoped_timer<type>(_timer_name);
    return f();
  }

private:
  std::string _timer_name;
};
} // namespace timer
} // namespace kaminpar
