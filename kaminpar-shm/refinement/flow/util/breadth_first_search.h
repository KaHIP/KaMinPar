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
  void reset() {
    _num_seeds = 0;
    _data.clear();
  }

  void add_seed(const NodeID seed) {
    _num_seeds += 1;
    _data.push_back(seed);
  }

  void add_seeds(const std::span<const NodeID> seeds) {
    _num_seeds += seeds.size();
    for (const NodeID seed : seeds) {
      _data.push_back(seed);
    }
  }

  template <typename Callback> void perform(Callback &&callback) {
    perform(0, std::forward<Callback>(callback));
  }

  template <typename Callback> void perform(const NodeID initial_distance, Callback &&callback) {
    constexpr bool kRequiresDistance =
        std::is_invocable_v<Callback, NodeID, NodeID, ScalableVector<NodeID> &>;

    _data.resize(_num_seeds);

    std::size_t begin = 0;
    std::size_t end = _data.size();

    NodeID distance = initial_distance;
    while (begin < end) {
      while (begin < end) {
        if constexpr (kRequiresDistance) {
          callback(_data[begin++], distance, _data);
        } else {
          callback(_data[begin++], _data);
        }
      }

      begin = end;
      end = _data.size();

      distance += 1;
    }
  }

private:
  std::size_t _num_seeds;
  ScalableVector<NodeID> _data;
};

class ParallelBFSRunner {
public:
  void reset(const NodeID max_num_nodes) {
    _num_seeds = 0;
    _data.clear();
    _data.reserve(max_num_nodes);
  }

  void add_seed(const NodeID seed) {
    _num_seeds += 1;
    _data.push_back(seed);
  }

  void add_seeds(const std::span<const NodeID> seeds) {
    _num_seeds += seeds.size();
    _data.push_back(seeds);
  }

  template <typename Callback> void perform(Callback &&callback) {
    constexpr bool kRequiresDistance =
        std::is_invocable_v<Callback, NodeID, NodeID, ScalableVector<NodeID> &>;

    _data.resize(_num_seeds);

    std::size_t begin = 0;
    std::size_t end = _data.size();

    std::size_t distance = 0;
    while (begin < end) {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(begin, end), [&](const auto &range) {
        BufferedVector<NodeID>::Buffer buffer = _data.local_buffer();

        for (std::size_t i = range.begin(), end = range.end(); i < end; ++i) {
          if constexpr (kRequiresDistance) {
            callback(_data[i], distance, buffer);
          } else {
            callback(_data[i], buffer);
          }
        }
      });

      _data.flush();

      begin = end;
      end = _data.size();

      distance += 1;
    }
  }

private:
  std::size_t _num_seeds;
  BufferedVector<NodeID> _data;
};

} // namespace kaminpar::shm
