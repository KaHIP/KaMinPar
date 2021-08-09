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
#define TIMER_DEFAULT kaminpar::timer::Type::DEFAULT
#define TIMER_BENCHMARK kaminpar::timer::Type::BENCHMARK

//
// Private helper macros
//
#define SCOPED_TIMER_IMPL2_3(name, description, line, type)                                                            \
  auto __SCOPED_TIMER__##line = (GLOBAL_TIMER.start_scoped_timer(name, description, type))
#define SCOPED_TIMER_IMPL1_3(name, description, line, type) SCOPED_TIMER_IMPL2_3(name, description, line, type)

#define SCOPED_TIMER_IMPL2_2(name, description_or_type, line)                                                          \
  auto __SCOPED_TIMER__##line = (GLOBAL_TIMER.start_scoped_timer(name, description_or_type))
#define SCOPED_TIMER_IMPL1_2(name, description_or_type, line) SCOPED_TIMER_IMPL2_2(name, description_or_type, line)

#ifdef KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER_3(name, description, type) SCOPED_TIMER_IMPL1_3(name, description, __LINE__, type)
#define START_TIMER_3(name, description, type) (GLOBAL_TIMER.start_timer(name, description, type))
#define SCOPED_TIMER_2(name, description_or_type) SCOPED_TIMER_IMPL1_2(name, description_or_type, __LINE__)
#define START_TIMER_2(name, description_or_type) (GLOBAL_TIMER.start_timer(name, description_or_type))
#define STOP_TIMER_1(type) (GLOBAL_TIMER.stop_timer(type))
#else // KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER_3(name, description, type)
#define START_TIMER_3(name, description, type)
#define SCOPED_TIMER_2(name, description_or_type)
#define START_TIMER_2(name, description_or_type)
#define STOP_TIMER_1(type)
#endif // KAMINPAR_ENABLE_TIMERS

#define SCOPED_TIMER_1(name) SCOPED_TIMER_2(name, TIMER_DEFAULT)
#define START_TIMER_1(name) START_TIMER_2(name, TIMER_DEFAULT)
#define STOP_TIMER_0() STOP_TIMER_1(TIMER_DEFAULT)

#define TIMED_SCOPE_3(name, description, type)                                                                         \
  kaminpar::timer::TimedScope<const std::string &>{GLOBAL_TIMER_PTR, name, description, type} + [&]
#define TIMED_SCOPE_2(name, description_or_type)                                                                       \
  kaminpar::timer::TimedScope<std::conditional_t<std::is_same_v<decltype(description_or_type), kaminpar::timer::Type>, \
                                                 const char *, const std::string &>>{GLOBAL_TIMER_PTR, name,           \
                                                                                     description_or_type} +            \
      [&]
#define TIMED_SCOPE_1(name) TIMED_SCOPE_2(name, TIMER_DEFAULT)

#define VARARG_SELECT_HELPER3(X, Y, Z, W, FUNC, ...) FUNC
#define VARARG_SELECT_HELPER1(X, Y, FUNC, ...) FUNC

//
// Public macro interface
//
#define ENABLE_TIMERS() (GLOBAL_TIMER.enable_all())
#define DISABLE_TIMERS() (GLOBAL_TIMER.disable_all())

#define SCOPED_TIMER(...)                                                                                              \
  VARARG_SELECT_HELPER3(, ##__VA_ARGS__, SCOPED_TIMER_3(__VA_ARGS__), SCOPED_TIMER_2(__VA_ARGS__),                     \
                        SCOPED_TIMER_1(__VA_ARGS__), ignore)
#define START_TIMER(...)                                                                                               \
  VARARG_SELECT_HELPER3(, ##__VA_ARGS__, START_TIMER_3(__VA_ARGS__), START_TIMER_2(__VA_ARGS__),                       \
                        START_TIMER_1(__VA_ARGS__), ignore)
#define STOP_TIMER(...)                                                                                                \
  VARARG_SELECT_HELPER1(, ##__VA_ARGS__, STOP_TIMER_1(__VA_ARGS__), STOP_TIMER_0(__VA_ARGS__), ignore)

// must be followed by a lambda body that may or may not return some value
#define TIMED_SCOPE(...)                                                                                               \
  VARARG_SELECT_HELPER3(, ##__VA_ARGS__, TIMED_SCOPE_3(__VA_ARGS__), TIMED_SCOPE_2(__VA_ARGS__),                       \
                        TIMED_SCOPE_1(__VA_ARGS__), ignore)

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
  static constexpr std::size_t kSpaceBetweenRestartsAndAnnotation = 1;
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

  struct TimerTreeNode {
    std::string_view name;
    std::string description;

    std::size_t restarts{0};
    Duration elapsed{};
    TimePoint start{};

    TimerTreeNode *parent{nullptr};
    std::map<std::string_view, TimerTreeNode *> children_tbl{};
    std::vector<std::unique_ptr<TimerTreeNode>> children;

    std::string annotation{};

    [[nodiscard]] std::string build_display_name_mr() const;
    [[nodiscard]] std::string build_display_name_hr() const;
  };

  struct TimerTree {
    TimerTreeNode root{};
    TimerTreeNode *current{&root};
  };

  template<typename StrType>
  bool is_empty_description(StrType description) {
    if constexpr (std::is_same_v<StrType, const char *>) {
      return *description == 0;
    } else {
      static_assert(std::is_same_v<std::decay_t<StrType>, std::string>);
      return description.empty();
    }
  }

public:
  static Timer &global();

  explicit Timer(std::string_view name);

  void start_timer(std::string_view name, const Type type = Type::DEFAULT) {
    start_timer<const char *>(name, "", type);
  }

  void start_timer(std::string_view name, const std::string &description) {
    start_timer<const std::string &>(name, description, Type::DEFAULT);
  }

  template<typename StrType>
  void start_timer(std::string_view name, StrType description, const Type type) {
    if (!_enabled[type]) { return; }
    std::lock_guard<std::mutex> lg{_mutex};

    // create new tree node if timer does not already exist
    const bool empty_description = is_empty_description(description);
    if (!empty_description || !_tree.current->children_tbl.contains(name)) {
      // create new tree node
      _tree.current->children.emplace_back(new TimerTreeNode{});
      auto *child = _tree.current->children.back().get();
      if (empty_description) { _tree.current->children_tbl[name] = child; }

      // init new tree node
      child->parent = _tree.current;
      child->name = name;
      child->description = description;

      // set as current node
      _tree.current = child;
    } else {
      _tree.current = _tree.current->children_tbl[name];
    }

    // update current timer
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

  template<typename StrType>
  auto start_scoped_timer(const std::string_view name, StrType description, const Type type) {
    start_timer(name, description, type);
    return timer::ScopedTimer{this, type};
  }

  decltype(auto) start_scoped_timer(const std::string_view name, const std::string &description) {
    return start_scoped_timer<const std::string &>(name, description, Type::DEFAULT);
  }

  decltype(auto) start_scoped_timer(const std::string_view name, const Type type) {
    return start_scoped_timer<const char *>(name, "", type);
  }

  void print_machine_readable(std::ostream &out);
  void print_human_readable(std::ostream &out);

  void enable(Type type = Type::DEFAULT) { _enabled[type] = true; }
  void disable(Type type = Type::DEFAULT) { _enabled[type] = false; }

  void enable_all() { std::fill(std::begin(_enabled), std::end(_enabled), true); }
  void disable_all() { std::fill(std::begin(_enabled), std::end(_enabled), false); }

private:
  void print_padded_timing(std::ostream &out, std::size_t start_col, const TimerTreeNode *node) const;

  void print_children_hr(std::ostream &out, const std::string &base_prefix, const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_time_col(std::size_t parent_prefix_len, const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_time_len(const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_restarts_len(const TimerTreeNode *node) const;

  void print_node_mr(std::ostream &out, const std::string &prefix, const TimerTreeNode *node);

  std::string_view _name;
  TimerTree _tree{};
  std::mutex _mutex{};
  bool _enabled[Type::NUM_TIMER_TYPES];

  std::size_t _hr_time_col;
  std::size_t _hr_max_time_len;
  std::size_t _hr_max_restarts_len;
};

namespace timer {
ScopedTimer::~ScopedTimer() { _timer->stop_timer(_type); }

template<typename StrType>
class TimedScope {
public:
  TimedScope(Timer *timer, std::string_view name, StrType description, Type type)
      : _timer{timer},
        _name{name},
        _description{description},
        _type{type} {}

  explicit TimedScope(Timer *timer, std::string_view name, StrType description)
      : TimedScope{timer, name, description, Type::DEFAULT} {}

  explicit TimedScope(Timer *timer, std::string_view name, Type type) : TimedScope{timer, name, "", type} {}

  template<typename F>
  decltype(auto) operator+(F &&f) {
    const auto scope = _timer->start_scoped_timer<StrType>(_name, _description, _type);
    return f();
  }

private:
  Timer *_timer;
  std::string_view _name;
  StrType _description;
  Type _type;
};
} // namespace timer
} // namespace kaminpar
