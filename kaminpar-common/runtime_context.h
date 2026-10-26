#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar {

// A separate pool for every partitioner. Engines never enter Random's legacy registry.
class RandomStreams {
public:
  explicit RandomStreams(int seed = 0)
      : _seed(seed),
        _engines([this] {
          const auto stream = _next_stream.fetch_add(1, std::memory_order_relaxed);
          return Random(
              static_cast<int>(static_cast<std::uint32_t>(_seed) + stream),
              random::thread_independent_seeding
          );
        }) {}

  Random &local() {
    return _engines.local();
  }
  int seed() const {
    return _seed;
  }

  // Requires all users to have completed; invalidates cached engine references.
  void reseed(int seed) {
    _engines.clear();
    _seed = seed;
    _next_stream = 0;
  }

private:
  int _seed;
  std::atomic<std::uint32_t> _next_stream{0};
  tbb::enumerable_thread_specific<Random> _engines;
};

// Mutable services belong to one library instance, with a stable address across moves.
class RuntimeContext {
public:
  explicit RuntimeContext(int seed = 0, int num_threads = tbb::task_arena::automatic)
      : random(seed),
        arena(num_threads) {
    arena.initialize();
  }

  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext &operator=(const RuntimeContext &) = delete;

  void begin_partition(int seed) {
    if (_finished)
      timer.reset();
    random.reseed(seed);
    _finished = false;
  }

  void finish_partition() {
    _finished = true;
  }

  void reset_after_failure() {
    timer.reset();
    _finished = false;
  }

  Timer timer{"KaMinPar"};
  RandomStreams random;
  std::atomic<bool> quiet{false};
  tbb::task_arena arena;

private:
  bool _finished = false;
};

} // namespace kaminpar
