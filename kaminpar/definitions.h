/*******************************************************************************
 * @file:   definitions.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  General type and macro definitions.
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>

#include "kaminpar/utils/logger.h"

#if __has_include(<sched.h>)
    #include <sched.h>
    #define HAS_SCHED_GETCPU
#endif

namespace kaminpar {
// Additional assertion levels to be used with KASSERT()
namespace assert {
#define ASSERTION_LEVEL_LIGHT 10
constexpr int light = ASSERTION_LEVEL_LIGHT;
#define ASSERTION_LEVEL_NORMAL 30
constexpr int normal = ASSERTION_LEVEL_NORMAL; // same value as defined in KASSERT
#define ASSERTION_LEVEL_HEAVY 40
constexpr int heavy = ASSERTION_LEVEL_HEAVY;
}; // namespace assert

struct Mandatory {};

#ifdef KAMINPAR_64BIT_NODE_IDS
using NodeID = uint64_t;
#else  // KAMINPAR_64BIT_NODE_IDS
using NodeID     = uint32_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
using EdgeID = uint64_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
using EdgeID     = uint32_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
using NodeWeight = int64_t;
using EdgeWeight = int64_t;
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = int32_t;
using EdgeWeight = int32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID     = uint32_t;
using BlockWeight = NodeWeight;
using Gain        = EdgeWeight;
using Degree      = EdgeID;
using Clustering  = std::vector<NodeID>;

constexpr BlockID     kInvalidBlockID     = std::numeric_limits<BlockID>::max();
constexpr NodeID      kInvalidNodeID      = std::numeric_limits<NodeID>::max();
constexpr EdgeID      kInvalidEdgeID      = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight  kInvalidNodeWeight  = std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight  kInvalidEdgeWeight  = std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight = std::numeric_limits<BlockWeight>::max();
constexpr Degree      kMaxDegree          = std::numeric_limits<Degree>::max();

template <typename T>
using scalable_vector = std::vector<T, tbb::scalable_allocator<T>>;
template <typename T>
using cache_aligned_vector = std::vector<T, tbb::cache_aligned_allocator<T>>;

namespace tag {
struct Parallel {};
constexpr inline Parallel par{};
struct Sequential {};
constexpr inline Sequential seq{};
} // namespace tag

// helper function to implement ASSERT() macros
namespace debug {
// helper function to implement ASSERT() and DBG() macros
template <bool abort_on_destruction>
class DisposableLogger {
public:
    template <typename... Args>
    explicit DisposableLogger(Args&&... args) : _logger(std::forward<Args>(args)...) {}

    ~DisposableLogger() {
        _logger << logger::RESET;
        _logger.flush();
        if constexpr (abort_on_destruction) {
            std::abort();
        }
    }

    template <typename Arg>
    DisposableLogger& operator<<(Arg&& arg) {
        _logger << std::forward<Arg>(arg);
        return *this;
    }

    // if the ASSERT or DBG macro is disabled, we use short circuit evaluation to dispose this logger and all calls to
    // it for do this, it must be implicitly convertible to bool (the return value does not matter)
    operator bool() {
        return false;
    } // NOLINT

private:
    Logger _logger;
};
} // namespace debug
} // namespace kaminpar

// clang-format off
#define FILENAME (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)
#define POSITION "[" << FILENAME << ":" << __LINE__ << "][" << __func__ << "]"
#ifdef HAS_SCHED_GETCPU
#define CPU "[CPU" << sched_getcpu() << "]"
#else // HAS_SCHED_GETCPU
#define CPU ""
#endif // HAS_SCHED_GETCPU
       
// Macros for debug output
//
// To use these macros, you must define a boolean variable kDebug somewhere
// The macros only produce output if the boolean variable is set to true
//
// DBG can be used just like LOG
// DBGC(cond) only produces output if the given condition evaluates to true
// IFDBG(expr) evaluates the expression and returns its result iff kDebug is set to true, otherwise returns the default value for its result data type
#define SET_DEBUG(value) [[maybe_unused]] static constexpr bool kDebug = value
#define DBGC(cond) (kDebug && (cond)) && kaminpar::debug::DisposableLogger<false>(std::cout) << kaminpar::logger::MAGENTA << POSITION << CPU << " " << kaminpar::logger::DEFAULT_TEXT
#define DBG DBGC(true)
#define IFDBG(expr) (kDebug ? (expr) : decltype(expr)())

// Macros for general console output
//
// LOG, SUCCESS, WARNING, ERROR print one line of colored output
// LOG: no colors
// SUCCESS: in green with prefix [Success]
// WARNING: in orange with prefix [Warning]
// ERROR: in red with prefix [Error]
//
// LLOG, LSUCCESS, LWARNING, LERROR print the message without appending a new line symbol
//
// FATAL_ERROR and FATAL_PERROR act like ERROR but also aborting the program after printing the message
// FATAL_PERROR appends the output of std::perror()
#define LOG_ERROR (kaminpar::Logger(std::cout) << kaminpar::logger::RED << "[Error] ")
#define LOG_LERROR (kaminpar::Logger(std::cout, "") << kaminpar::logger::RED)
#define LOG_SUCCESS (kaminpar::Logger(std::cout) << kaminpar::logger::GREEN << "[Success] ")
#define LOG_LSUCCESS (kaminpar::Logger(std::cout, "") << kaminpar::logger::GREEN)
#define LOG_WARNING (kaminpar::Logger(std::cout) << kaminpar::logger::ORANGE << "[Warning] ")
#define LOG_LWARNING (kaminpar::Logger(std::cout, "") << kaminpar::logger::ORANGE)
#define FATAL_ERROR (kaminpar::debug::DisposableLogger<true>(std::cout) << kaminpar::logger::RED << "[Fatal] ")
#define FATAL_PERROR (kaminpar::debug::DisposableLogger<true>(std::cout, std::string(": ") + std::strerror(errno) + "\n") << kaminpar::logger::RED << "[Fatal] ")

// Macro that evaluates to true or false depending on whether another macro is defined or undefined
// use DETECT_EXIST(SOME_OTHER_MACRO) to detect whether SOME_OTHER_MACRO is defined or undefined
//
// Copied from https://stackoverflow.com/questions/41265750/how-to-get-a-boolean-indicating-if-a-macro-is-defined-or-not
#define SECOND_ARG(A,B,...) B
#define CONCAT2(A,B) A ## B
#define DETECT_EXIST_TRUE ~,1
#define DETECT_EXIST_IMPL(...) SECOND_ARG(__VA_ARGS__)
#define DETECT_EXIST(X) DETECT_EXIST_IMPL(CONCAT2(DETECT_EXIST_TRUE,X), 0, ~)
// clang-format on

// Macros for console output
//
// V(x) prints x<space><value of x><space>, e.g., use LOG << V(a) << V(b) << V(c); to quickly print the values of
// variables a, b, c
// C(x, y) prints [<value of x> --> <value of y>]
#define V(x)    std::string(#x "=") << (x) << " "
#define C(x, y) "[" << (x) << " --> " << (y) << "] "

// Macros for statistics
//
// SET_STATISTICS(false or true): disable or enable statistics for the given module
// SET_STATISTICS_FROM_GLOBAL: respect compiler option -DKAMINPAR_ENABLE_STATISTICS for this module
// IFSTATS(x): only evaluate this expression if statistics are enabled
// STATS: LOG for statistics output: only evaluate and output if statistics are enabled
#define SET_STATISTICS(value) constexpr static bool kStatistics = value
#define IFSTATS(x)            (kStatistics ? (x) : std::decay_t<decltype(x)>())
#define STATS                 kStatistics&& kaminpar::debug::DisposableLogger<false>(std::cout) << kaminpar::logger::CYAN

#ifdef KAMINPAR_ENABLE_STATISTICS
    #define SET_STATISTICS_FROM_GLOBAL() SET_STATISTICS(true)
#else // KAMINPAR_ENABLE_STATISTICS
    #define SET_STATISTICS_FROM_GLOBAL() SET_STATISTICS(false)
#endif // KAMINPAR_ENABLE_STATISTISC
