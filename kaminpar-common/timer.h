/*******************************************************************************
 * Hierarchical timer for easy time measurements.
 *
 * @file:   timer.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#define GLOBAL_TIMER (kaminpar::Timer::global())
#define GLOBAL_TIMER_PTR &(GLOBAL_TIMER)

//
// Private helper macros
//
#define SCOPED_TIMER_IMPL2_2(name, description, line)                                              \
  auto __SCOPED_TIMER__##line = (GLOBAL_TIMER.start_scoped_timer(name, description))
#define SCOPED_TIMER_IMPL1_2(name, description, line) SCOPED_TIMER_IMPL2_2(name, description, line)

#ifdef KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER_2(name, description) SCOPED_TIMER_IMPL1_2(name, description, __LINE__)
#define START_TIMER_2(name, description) (GLOBAL_TIMER.start_timer(name, description))
#else // KAMINPAR_ENABLE_TIMERS
#define SCOPED_TIMER_2(name, description)
#define START_TIMER_2(name, description)
#endif // KAMINPAR_ENABLE_TIMERS

#define SCOPED_TIMER_1(name) SCOPED_TIMER_2(name, "")
#define START_TIMER_1(name) START_TIMER_2(name, "")

#define TIMED_SCOPE_2(name, description)                                                           \
  kaminpar::timer::TimedScope<const std::string &>{GLOBAL_TIMER_PTR, name, description} + [&]
#define TIMED_SCOPE_1(name) TIMED_SCOPE_2(name, "")

#define VARARG_SELECT_HELPER2(X, Y, Z, FUNC, ...) FUNC

//
// Public macro interface
//
#ifdef KAMINPAR_ENABLE_TIMERS
#define ENABLE_TIMERS() (GLOBAL_TIMER.enable())
#define DISABLE_TIMERS() (GLOBAL_TIMER.disable())
#else // KAMINPAR_ENABLE_TIMERS
#define ENABLE_TIMERS()
#define DISABLE_TIMERS()
#endif // KAMINPAR_ENABLE_TIMERS

#define SCOPED_TIMER(...)                                                                          \
  VARARG_SELECT_HELPER2(                                                                           \
      , ##__VA_ARGS__, SCOPED_TIMER_2(__VA_ARGS__), SCOPED_TIMER_1(__VA_ARGS__), ignore            \
  )
#define START_TIMER(...)                                                                           \
  VARARG_SELECT_HELPER2(                                                                           \
      , ##__VA_ARGS__, START_TIMER_2(__VA_ARGS__), START_TIMER_1(__VA_ARGS__), ignore              \
  )
#ifdef KAMINPAR_ENABLE_TIMERS
#define STOP_TIMER() (GLOBAL_TIMER.stop_timer())
#else // KAMINPAR_ENABLE_TIMERS
#define STOP_TIMER()
#endif // KAMINPAR_ENABLE_TIMERS

// must be followed by a lambda body that may or may not return some value
#define TIMED_SCOPE(...)                                                                           \
  VARARG_SELECT_HELPER2(                                                                           \
      , ##__VA_ARGS__, TIMED_SCOPE_2(__VA_ARGS__), TIMED_SCOPE_1(__VA_ARGS__), ignore              \
  )

namespace kaminpar {
class Timer;

namespace timer {
inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

class ScopedTimer {
public:
  explicit ScopedTimer(Timer *timer) : _timer(timer) {}

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

  ScopedTimer(ScopedTimer &&other) noexcept : _timer{other._timer} {
    other._timer = nullptr;
  };
  ScopedTimer &operator=(ScopedTimer &&other) noexcept {
    return (std::swap(_timer, other._timer), *this);
  };

  inline ~ScopedTimer();

private:
  Timer *_timer;
};
} // namespace timer

class Timer {
  static constexpr bool kDebug = false;

  static constexpr std::size_t kSpaceBetweenTimeAndRestarts = 1;
  static constexpr std::size_t kSpaceBetweenRestartsAndAnnotation = 1;
  static constexpr std::string_view kBranch = "|- ";
  static constexpr std::string_view kEdge = "|  ";
  static constexpr std::string_view kTailBranch = "`- ";
  static constexpr std::string_view kTailEdge = "   ";
  static constexpr std::string_view kNameDel = ": ";
  static constexpr char kPadding = '.';
  static constexpr std::string_view kSecondsUnit = " s";

public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using Duration = std::chrono::high_resolution_clock::duration;

  struct TimerTreeNode {
    std::string_view name;
    std::string description;

    std::size_t restarts = 0;
    Duration elapsed;
    TimePoint start;

    TimerTreeNode *parent = nullptr;
    std::map<std::string_view, TimerTreeNode *> children_tbl;
    std::vector<std::unique_ptr<TimerTreeNode>> children;

    std::string annotation;

    [[nodiscard]] std::string build_display_name_mr() const;
    [[nodiscard]] std::string build_display_name_hr() const;

    [[nodiscard]] inline double seconds() const {
      return static_cast<double>(
                 std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
             ) /
             1000.0;
    }
  };

  struct TimerTree {
    TimerTreeNode root;
    TimerTreeNode *current = &root;
  };

public:
  static Timer &global();

  explicit Timer(std::string_view name);

  void start_timer(std::string_view name) {
    start_timer<const char *>(name, "");
  }

  void start_timer(std::string_view name, const std::string &description) {
    start_timer<const std::string &>(name, description);
  }

  template <typename String> void start_timer(std::string_view name, String description) {
    std::lock_guard<std::mutex> lg(_mutex);
    if (_disabled > 0) {
      return;
    }

    // create new tree node if timer does not already exist
    auto tbl_contains = [&](std::string_view name) {
      return _tree.current->children_tbl.find(name) != _tree.current->children_tbl.end();
    };
    const bool empty_description = is_empty_description(description);
    if (!empty_description || !tbl_contains(name)) {
      // create new tree node
      _tree.current->children.emplace_back(new TimerTreeNode{});
      auto *child = _tree.current->children.back().get();
      if (empty_description) {
        _tree.current->children_tbl[name] = child;
      }

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
    start_timer_impl();
  }

  void stop_timer() {
    std::lock_guard<std::mutex> lg{_mutex};

    if (_disabled > 0) {
      return;
    }

    stop_timer_impl();
    _tree.current = _tree.current->parent;
  }

  template <typename String>
  auto start_scoped_timer(const std::string_view name, String description) {
    start_timer(name, description);
    return timer::ScopedTimer{this};
  }

  decltype(auto) start_scoped_timer(const std::string_view name, const std::string &description) {
    return start_scoped_timer<const std::string &>(name, description);
  }

  decltype(auto) start_scoped_timer(const std::string_view name) {
    return start_scoped_timer<const char *>(name, "");
  }

  void enable() {
    std::lock_guard<std::mutex> lg(_mutex);
    _disabled = std::max(0, _disabled - 1);
  }

  void disable() {
    std::lock_guard<std::mutex> lg(_mutex);
    _disabled++;
  }

  void annotate(std::string annotation) {
    _annotation = std::move(annotation);
  }

  [[nodiscard]] inline TimerTreeNode &tree() {
    return _tree.root;
  }

  [[nodiscard]] inline const TimerTreeNode &tree() const {
    return _tree.root;
  }

  [[nodiscard]] inline double elapsed_seconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(timer::now() - _tree.root.start)
               .count() /
           1000.0;
  }

  void print_machine_readable(std::ostream &out, int max_depth = std::numeric_limits<int>::max());

  void print_human_readable(std::ostream &out, int max_depth = std::numeric_limits<int>::max());

  void reset();

private:
  template <typename String> bool is_empty_description(String description) {
    if constexpr (std::is_same_v<String, const char *>) {
      return *description == 0;
    } else {
      return description.empty();
    }
  }

  void start_timer_impl();
  void stop_timer_impl();

  [[nodiscard]] std::size_t
  compute_time_col(std::size_t parent_prefix_len, const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_time_len(const TimerTreeNode *node) const;

  [[nodiscard]] std::size_t compute_restarts_len(const TimerTreeNode *node) const;

  void
  print_padded_timing(std::ostream &out, std::size_t start_col, const TimerTreeNode *node) const;

  void print_children_hr(
      std::ostream &out, const std::string &base_prefix, const TimerTreeNode *node, int max_depth
  ) const;

  void print_node_mr(
      std::ostream &out, const std::string &prefix, const TimerTreeNode *node, int max_depth
  ) const;

  std::string_view _name;
  std::string _annotation;
  TimerTree _tree;
  std::mutex _mutex;
  int _disabled = 0;

  std::size_t _hr_time_col = 0;
  std::size_t _hr_max_time_len = 0;
  std::size_t _hr_max_restarts_len = 0;
};

namespace timer {
ScopedTimer::~ScopedTimer() {
  _timer->stop_timer();
}

template <typename String> class TimedScope {
public:
  TimedScope(Timer *timer, std::string_view name, String description)
      : _timer(timer),
        _name(name),
        _description(description) {}

  explicit TimedScope(Timer *timer, std::string_view name) : TimedScope(timer, name, "") {}

  template <typename F> decltype(auto) operator+(F &&f) {
#ifdef KAMINPAR_ENABLE_TIMERS
    const auto scope = _timer->start_scoped_timer<String>(_name, _description);
#endif // KAMINPAR_ENABLE_TIMERS
    return f();
  }

private:
  Timer *_timer;
  std::string_view _name;
  String _description;
};
} // namespace timer
} // namespace kaminpar
