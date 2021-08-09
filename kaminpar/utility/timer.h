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

#define GLOBAL_TIMER (kaminpar::Timer::global())
#define GLOBAL_TIMER_PTR &(GLOBAL_TIMER)

#define ENABLE_TIMERS() (GLOBAL_TIMER.enable_all())
#define DISABLE_TIMERS() (GLOBAL_TIMER.disable_all())

#define TIMER_DEFAULT timer::Type::DEFAULT
#define TIMER_BENCHMARK timer::Type::BENCHMARK

// Helper macros
#define SCOPED_TIMER_IMPL2(x, line, func)                                                                              \
auto __SCOPED_TIMER__##line = (kaminpar::Timer::global().start_scoped_timer(x, "", func))
#define SCOPED_TIMER_IMPL1(x, line, func) SCOPED_TIMER_IMPL2(x, line, func)

// Macro interface
#ifdef KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER(x) SCOPED_TIMER_IMPL1(x, __LINE__, TIMER_DEFAULT)
#define START_TIMER(x) (kaminpar::Timer::global().start_timer(x))
#define STOP_TIMER() (kaminpar::Timer::global().stop_timer())
#else // KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER(x)
#define START_TIMER(x)
#define STOP_TIMER()
#endif // KAMINPAR_ENABLE_TIMERS

// must be followed by a lambda body that may or may not return some value
#define TIMED_SCOPE(x) kaminpar::timer::TimedScope{GLOBAL_TIMER_PTR, x} + [&]

namespace kaminpar {
class Timer;

namespace timer {
inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

enum Type {
  DEFAULT,
  BENCHMARK,
  NUM_TIMER_TYPES,
};

class ScopedTimer {
public:
  explicit ScopedTimer(Timer *timer, const Type type) : _timer{timer}, _type{type} {}
  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
  ScopedTimer(ScopedTimer &&other) noexcept : _timer{other._timer} { other._timer = nullptr; };
  ScopedTimer &operator=(ScopedTimer &&other) noexcept { return (std::swap(_timer, other._timer), *this); };
  inline ~ScopedTimer();

private:
  Timer *_timer;
  Type _type;
};
} // namespace timer

class Timer {
  using Type = timer::Type;

  static constexpr bool kDebug = false;

  static constexpr std::size_t kSpaceBetweenTimeAndRestarts = 1;
  static constexpr std::string_view kBranch = "|-- ";
  static constexpr std::string_view kEdge = "|   ";
  static constexpr std::string_view kTailBranch = "`-- ";
  static constexpr std::string_view kTailEdge = "    ";
  static constexpr std::string_view kNameDel = ": ";
  static constexpr char kPadding = '.';
  static constexpr std::string_view kSecondsUnit = " s";

public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using Duration = std::chrono::high_resolution_clock::duration;

private:
  struct TimerTreeNode {
    std::string_view name;
    std::string description;

    std::size_t restarts{0};
    Duration elapsed{};
    TimePoint start{};

    TimerTreeNode *parent{nullptr};
    std::map<std::string_view, std::unique_ptr<TimerTreeNode>> children{};

    [[nodiscard]] std::string build_display_name_mr() const;
    [[nodiscard]] std::string build_display_name_hr() const;
  };

  struct TimerTree {
    TimerTreeNode root{};
    TimerTreeNode *current{&root};
  };

public:
  static Timer &global();

  explicit Timer(std::string_view name);

  void start_timer(std::string_view name, const std::string &description = "", const Type type = Type::DEFAULT) {
    if (!_enabled[type]) { return; }
    std::lock_guard<std::mutex> lg{_mutex};

    // create new tree node if timer does not already exist
    auto *current = _tree.current;
    if (!current->children.contains(name)) {
      auto node = std::make_unique<TimerTreeNode>();
      node->parent = current;
      node->name = name;
      node->description = description;
      current->children[name] = std::move(node);
    }

    // update current timer
    _tree.current = current->children[name].get();
    ++_tree.current->restarts;
    _tree.current->start = timer::now();
  }

  void stop_timer(const Type type = Type::DEFAULT) {
    if (!_enabled[type]) { return; }
    std::lock_guard<std::mutex> lg{_mutex};

    const TimePoint end = timer::now();
    _tree.current->elapsed += end - _tree.current->start;
    _tree.current = _tree.current->parent;
  }

  timer::ScopedTimer start_scoped_timer(const std::string_view name, const std::string &description = "",
                                        const Type type = Type::DEFAULT) {
    start_timer(name, description, type);
    return timer::ScopedTimer{this, type};
  }

  void print_machine_readable(std::ostream &out);
  void print_human_readable(std::ostream &out);

  void enable(Type type = Type::DEFAULT) { _enabled[type] = true; }
  void disable(Type type = Type::DEFAULT) { _enabled[type] = false; }

  void enable_all() { std::fill(std::begin(_enabled), std::end(_enabled), true); }
  void disable_all() { std::fill(std::begin(_enabled), std::end(_enabled), false); }

private:
  static void print_padded_timing(std::ostream &out, std::size_t start_col, std::size_t time_col,
                                  std::size_t max_time_len, const TimerTreeNode *node);

  void print_children_hr(std::ostream &out, const std::string &base_prefix, const TimerTreeNode *node, size_t time_col,
                         size_t max_time_len) const;

  [[nodiscard]] std::size_t compute_time_col(std::size_t parent_prefix_len, const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_time_len(const TimerTreeNode *node) const;

  void print_node_mr(std::ostream &out, const std::string &prefix, const TimerTreeNode *node);

  std::string_view _name;
  TimerTree _tree{};
  std::mutex _mutex{};
  bool _enabled[Type::NUM_TIMER_TYPES];
};

namespace timer {
ScopedTimer::~ScopedTimer() { _timer->stop_timer(_type); }

class TimedScope {
public:
  explicit TimedScope(Timer *timer, std::string_view name, const std::string &description = "",
                      Type type = Type::DEFAULT)
      : _timer{timer},
        _name{name},
        _description{description},
        _type{type} {}

  template<typename F>
  decltype(auto) operator+(F &&f) {
    const auto scope = _timer->start_scoped_timer(_name, _description, _type);
    return f();
  }

private:
  Timer *_timer;
  std::string_view _name;
  const std::string &_description;
  Type _type;
};
} // namespace timer
} // namespace kaminpar
