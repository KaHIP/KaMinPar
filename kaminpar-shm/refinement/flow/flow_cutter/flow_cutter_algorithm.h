#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"

namespace kaminpar::shm {

class FlowCutterAlgorithm {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  struct Move {
    NodeID node;
    BlockID old_block;
    BlockID new_block;
  };

  struct Result {
    bool time_limit_exceeded;

    EdgeWeight gain;
    bool improved_balance;
    std::span<Move> moves;

    Result(bool time_limit_exceeded = false)
        : time_limit_exceeded(time_limit_exceeded),
          gain(0),
          improved_balance(false) {};

    Result(EdgeWeight gain, bool improved_balance, std::span<Move> moves)
        : time_limit_exceeded(false),
          gain(gain),
          improved_balance(improved_balance),
          moves(moves) {};

    [[nodiscard]] static Result empty() {
      return Result(0, false, {});
    }

    [[nodiscard]] static Result time_limit() {
      return Result(true);
    }
  };

public:
  FlowCutterAlgorithm() : _has_time_limit(false) {}

  virtual ~FlowCutterAlgorithm() = default;

  [[nodiscard]] virtual Result compute_cut(
      const BorderRegion &border_region, const FlowNetwork &flow_network, bool run_sequentially
  ) = 0;

  virtual void free() = 0;

  void set_time_limit(const std::size_t time_limit, TimePoint start_time) {
    _has_time_limit = time_limit != std::numeric_limits<std::size_t>::max();
    _time_limit = time_limit;
    _start_time = start_time;
  }

protected:
  [[nodiscard]] bool time_limit_exceeded() const {
    if (_has_time_limit) {
      using namespace std::chrono;

      TimePoint current_time = Clock::now();
      std::size_t time_elapsed = duration_cast<milliseconds>(current_time - _start_time).count();

      return time_elapsed >= _time_limit * 60 * 1000;
    }

    return false;
  }

private:
  bool _has_time_limit;
  std::size_t _time_limit;
  TimePoint _start_time;
};

} // namespace kaminpar::shm
