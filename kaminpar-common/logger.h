/*******************************************************************************
 * Helper class for console logging.
 *
 * @file:   logger.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/spin_mutex.h>

#include "kaminpar-common/parallel/sched_getcpu.h"

// Macros for debug output
//
// To use these macros, you must define a boolean variable kDebug somewhere
// The macros only produce output if the boolean variable is set to true
//
// DBG can be used just like LOG
// DBGC(cond) only produces output if the given condition evaluates to true
// IFDBG(expr) evaluates the expression and returns its result iff kDebug is set
// to true, otherwise returns the default value for its result data type
#define FILENAME (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)
#define POSITION FILENAME << ":" << __LINE__ << "(" << __func__ << ")"
#ifdef HAS_SCHED_GETCPU
// #define CPU "[CPU" << sched_getcpu() << "]"
#define CPU ""
#else // HAS_SCHED_GETCPU
#define CPU ""
#endif // HAS_SCHED_GETCPU

#define SET_DEBUG(value) [[maybe_unused]] static constexpr bool kDebug = value
#define DBGC(cond)                                                                                 \
  (kDebug && (cond)) && kaminpar::DisposableLogger<false>(std::cout)                               \
                            << kaminpar::logger::MAGENTA << POSITION << CPU << " "                 \
                            << kaminpar::logger::DEFAULT_TEXT
#define DBG DBGC(true)
#define IFDBG(x) (kDebug ? (x) : std::decay_t<decltype(x)>())
#define IF_DBG if constexpr (kDebug)

// Macros for general console output
//
// LOG, SUCCESS, WARNING, ERROR print one line of colored output
// LOG: no colors
// SUCCESS: in green with prefix [Success]
// WARNING: in orange with prefix [Warning]
// ERROR: in red with prefix [Error]
//
// LLOG, LSUCCESS, LWARNING, LERROR print the message without appending a new
// line symbol
//
// FATAL_ERROR and FATAL_PERROR act like ERROR but also aborting the program
// after printing the message FATAL_PERROR appends the output of std::perror()
#define LOG (kaminpar::Logger())
#define LLOG (kaminpar::Logger(std::cout, ""))

#define LOG_ERROR (kaminpar::Logger(std::cout) << kaminpar::logger::RED << "[Error] ")
#define LOG_LERROR (kaminpar::Logger(std::cout, "") << kaminpar::logger::RED)
#define LOG_SUCCESS (kaminpar::Logger(std::cout) << kaminpar::logger::GREEN << "[Success] ")
#define LOG_LSUCCESS (kaminpar::Logger(std::cout, "") << kaminpar::logger::GREEN)
#define LOG_WARNING (kaminpar::Logger(std::cout) << kaminpar::logger::ORANGE << "[Warning] ")
#define LOG_LWARNING (kaminpar::Logger(std::cout, "") << kaminpar::logger::ORANGE)
#define FATAL_ERROR                                                                                \
  (kaminpar::DisposableLogger<true>(std::cout) << kaminpar::logger::RED << "[Fatal] ")
#define FATAL_PERROR                                                                               \
  (kaminpar::DisposableLogger<true>(std::cout, std::string(": ") + std::strerror(errno) + "\n")    \
   << kaminpar::logger::RED << "[Fatal] ")

// V(x) prints x<space><value of x><space>, e.g., use LOG << V(a) << V(b) <<
// V(c); to quickly print the values of variables a, b, c C(x, y) prints [<value
// of x> --> <value of y>]
#define V(x) std::string(#x "=") << (x) << " "
#define C(x, y) "[" << (x) << " --> " << (y) << "] "

// Macros for statistics
//
// SET_STATISTICS(false or true): disable or enable statistics for the given
// module SET_STATISTICS_FROM_GLOBAL: respect compiler option
// -DKAMINPAR_ENABLE_STATISTICS for this module IFSTATS(x): only evaluate this
// expression if statistics are enabled STATS: LOG for statistics output: only
// evaluate and output if statistics are enabled
#define SET_STATISTICS(value) [[maybe_unused]] constexpr static bool kStatistics = value
#define IFSTATSC(cond, expr) ((cond) ? (expr) : std::decay_t<decltype(expr)>())
#define IFSTATS(expr) IFSTATSC(kStatistics, expr)
#define IF_STATSC(cond) if ((cond))
#define IF_STATS if constexpr (kStatistics)
#define LOG_STATS                                                                                  \
  kaminpar::DisposableLogger<false>(std::cout) << kaminpar::logger::CYAN << "[Statistics] "
#define STATSC(cond) ((cond)) && LOG_STATS
#define STATS STATSC(kStatistics)

#ifdef KAMINPAR_ENABLE_STATISTICS
#define SET_STATISTICS_FROM_GLOBAL() SET_STATISTICS(true)
#else // KAMINPAR_ENABLE_STATISTICS
#define SET_STATISTICS_FROM_GLOBAL() SET_STATISTICS(false)
#endif // KAMINPAR_ENABLE_STATISTISC

namespace kaminpar {
namespace logger {
template <typename T, typename = void> struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<
    T,
    std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_container_v = !std::is_same_v<std::decay_t<T>, std::string> &&  //
                                !std::is_same_v<std::decay_t<T>, char *> &&       //
                                !std::is_same_v<std::decay_t<T>, const char *> && //
                                is_iterable<T>::value;

class ContainerFormatter {
public:
  virtual ~ContainerFormatter() = default;
  virtual void print(const std::vector<std::string> &container, std::ostream &out) const = 0;
};

class CompactContainerFormatter : public ContainerFormatter {
public:
  constexpr explicit CompactContainerFormatter(std::string_view sep) noexcept : _sep{sep} {}
  void print(const std::vector<std::string> &container, std::ostream &out) const final;

private:
  std::string_view _sep;
};

class Table : public ContainerFormatter {
public:
  constexpr explicit Table(std::size_t width) noexcept : _width{width} {}
  void print(const std::vector<std::string> &container, std::ostream &out) const final;

private:
  std::size_t _width;
};

class TextFormatter {
public:
  virtual ~TextFormatter() = default;
  virtual void print(const std::string &text, std::ostream &out) const = 0;
};

class DefaultTextFormatter final : public TextFormatter {
public:
  void print(const std::string &text, std::ostream &out) const final;
};

class Colorized : public TextFormatter {
public:
  enum class Color {
    RED,
    GREEN,
    MAGENTA,
    ORANGE,
    CYAN,
    RESET
  };

  constexpr explicit Colorized(Color color) noexcept : _color{color} {}
  void print(const std::string &text, std::ostream &out) const final;

private:
  Color _color;
};

template <typename T>
constexpr bool is_text_formatter_v = std::is_base_of_v<TextFormatter, std::decay_t<T>>;

template <typename T>
constexpr bool is_container_formatter_v = std::is_base_of_v<ContainerFormatter, std::decay_t<T>>;

template <typename T>
constexpr bool is_default_log_arg_v =
    !is_container_v<T> && !is_text_formatter_v<T> && !is_container_formatter_v<T>;

extern DefaultTextFormatter DEFAULT_TEXT;
extern Colorized RED;
extern Colorized GREEN;
extern Colorized MAGENTA;
extern Colorized ORANGE;
extern Colorized CYAN;
extern Colorized RESET;
extern CompactContainerFormatter DEFAULT_CONTAINER;
extern CompactContainerFormatter COMPACT;
extern Table TABLE;
} // namespace logger

class Logger {
public:
  Logger();
  explicit Logger(std::ostream &out, std::string append = "\n");

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) noexcept = default;
  Logger &operator=(Logger &&) = delete;
  virtual ~Logger() {
    flush();
  };

  template <typename Arg, std::enable_if_t<logger::is_default_log_arg_v<Arg>, bool> = true>
  Logger &operator<<(Arg &&arg) {
    std::stringstream ss;
    ss << arg;
    _text_formatter->print(ss.str(), _buffer);
    return *this;
  }

  template <
      typename Formatter,
      std::enable_if_t<logger::is_text_formatter_v<Formatter>, bool> = true>
  Logger &operator<<(Formatter &&formatter) {
    _text_formatter = std::make_unique<std::decay_t<Formatter>>(formatter);
    return *this;
  }

  template <
      typename Formatter,
      std::enable_if_t<logger::is_container_formatter_v<Formatter>, bool> = true>
  Logger &operator<<(Formatter &&formatter) {
    _container_formatter = std::make_unique<std::decay_t<Formatter>>(formatter);
    return *this;
  }

  template <typename T, std::enable_if_t<logger::is_container_v<T>, bool> = true>
  Logger &operator<<(T &&container) {
    std::vector<std::string> str;
    for (const auto &element : container) {
      std::stringstream ss;
      ss << element;
      str.push_back(ss.str());
    }
    _container_formatter->print(str, _buffer);
    return *this;
  }

  template <typename K, typename V> Logger &operator<<(const std::pair<K, V> &&pair) {
    (*this) << "<" << pair.first << ", " << pair.second << ">";
    return *this;
  }

  void flush();

  static void set_quiet_mode(bool quiet);
  static bool is_quiet();

private:
  static std::atomic<std::uint8_t> _quiet;

  static tbb::spin_mutex &flush_mutex();

  std::unique_ptr<logger::TextFormatter> _text_formatter{
      std::make_unique<std::decay_t<decltype(logger::DEFAULT_TEXT)>>(logger::DEFAULT_TEXT)
  };
  std::unique_ptr<logger::ContainerFormatter> _container_formatter{
      std::make_unique<std::decay_t<decltype(logger::DEFAULT_CONTAINER)>>(logger::DEFAULT_CONTAINER)
  };

  std::ostringstream _buffer;
  std::ostream &_out;
  std::string _append;
  bool _flushed{false};
};

// Helper function to implement ASSERT() and DBG() macros
template <bool abort_on_destruction> class DisposableLogger {
public:
  template <typename... Args>
  explicit DisposableLogger(Args &&...args) : _logger(std::forward<Args>(args)...) {}

  ~DisposableLogger() {
    _logger << logger::RESET;
    _logger.flush();
    if constexpr (abort_on_destruction) {
      std::abort();
    }
  }

  template <typename Arg> DisposableLogger &operator<<(Arg &&arg) {
    _logger << std::forward<Arg>(arg);
    return *this;
  }

  // if the ASSERT or DBG macro is disabled, we use short circuit evaluation to
  // dispose this logger and all calls to it for do this, it must be implicitly
  // convertible to bool (the return value does not matter)
  operator bool() {
    return false;
  } // NOLINT

private:
  Logger _logger;
};
} // namespace kaminpar
