#pragma once

#include <span>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/buffered_vector.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

class BFSRunner {
public:
  using Queue = ScalableVector<NodeID>;

  BFSRunner() = default;

  BFSRunner(BFSRunner &&) noexcept = default;
  BFSRunner &operator=(BFSRunner &&) noexcept = default;

  BFSRunner(const BFSRunner &) = delete;
  BFSRunner &operator=(const BFSRunner &) = delete;

  void reset() {
    _queue.clear();
  }

  void add_seed(const NodeID seed) {
    _queue.push_back(seed);
  }

  void add_seeds(const std::span<const NodeID> seeds) {
    for (const NodeID seed : seeds) {
      _queue.push_back(seed);
    }
  }

  template <typename Callback> void perform(Callback &&callback) {
    perform(0, std::forward<Callback>(callback));
  }

  template <typename Callback> void perform(const NodeID initial_distance, Callback &&callback) {
    constexpr bool kRequiresDistance = std::is_invocable_v<Callback, NodeID, NodeID, Queue &>;

    const std::size_t num_seeds = _queue.size();

    std::size_t begin = 0;
    std::size_t end = num_seeds;

    NodeID distance = initial_distance;
    while (begin < end) {
      while (begin < end) {
        if constexpr (kRequiresDistance) {
          callback(_queue[begin++], distance, _queue);
        } else {
          callback(_queue[begin++], _queue);
        }
      }

      begin = end;
      end = _queue.size();

      distance += 1;
    }

    _queue.resize(num_seeds);
  }

  void free() {
    _queue.clear();
    _queue.shrink_to_fit();
  }

private:
  Queue _queue;
};

class ParallelBFSRunner {
public:
  using Queue = BufferedVector<NodeID>::Buffer;

  ParallelBFSRunner() = default;

  ParallelBFSRunner(ParallelBFSRunner &&) noexcept = default;
  ParallelBFSRunner &operator=(ParallelBFSRunner &&) noexcept = default;

  ParallelBFSRunner(const ParallelBFSRunner &) = delete;
  ParallelBFSRunner &operator=(const ParallelBFSRunner &) = delete;

  void reset(const NodeID max_num_nodes) {
    _queue.clear();
    _queue.reserve(max_num_nodes);
  }

  void add_seed(const NodeID seed) {
    _queue.push_back(seed);
  }

  void add_seeds(const std::span<const NodeID> seeds) {
    _queue.push_back(seeds);
  }

  template <typename Callback> void perform(Callback &&callback) {
    perform(0, std::forward<Callback>(callback));
  }

  template <typename Callback> void perform(const NodeID initial_distance, Callback &&callback) {
    constexpr bool kRequiresDistance = std::is_invocable_v<Callback, NodeID, NodeID, Queue>;

    const std::size_t num_seeds = _queue.size();

    std::size_t begin = 0;
    std::size_t end = num_seeds;

    std::size_t distance = initial_distance;
    while (begin < end) {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(begin, end), [&](const auto &range) {
        Queue local_queue = _queue.local_buffer();

        for (std::size_t i = range.begin(), end = range.end(); i < end; ++i) {
          if constexpr (kRequiresDistance) {
            callback(_queue[i], distance, local_queue);
          } else {
            callback(_queue[i], local_queue);
          }
        }
      });

      _queue.flush();

      begin = end;
      end = _queue.size();

      distance += 1;
    }

    _queue.resize(num_seeds);
  }

  void free() {
    _queue.free();
  }

private:
  BufferedVector<NodeID> _queue;
};

} // namespace kaminpar::shm
