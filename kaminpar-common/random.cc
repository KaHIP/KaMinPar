/*******************************************************************************
 * Thread-local singleton for PRNG.
 *
 * @file:   random.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/random.h"

#include <mutex>

#include <tbb/task_arena.h>

namespace kaminpar {
std::mutex Random::_create_mutex;
std::vector<std::unique_ptr<Random>> Random::_instances;
int Random::_seed = 0;

Random &Random::instance() {
  thread_local static Random &instance = create_instance();
  return instance;
}

Random &Random::create_instance() {
  std::lock_guard<std::mutex> lock(_create_mutex);
  _instances.push_back(std::make_unique<Random>());
  return *_instances.back();
}

void Random::reseed(const int seed) {
  _seed = seed;
  for (auto &instance : _instances) {
    instance->reinit(seed);
  }
}

int Random::get_seed() {
  return _seed;
}

Random::Random()
    : _generator(_seed + tbb::this_task_arena::current_thread_index()),
      _bool_dist(0, 1),
      _real_dist(0, 1),
      _next_random_bool(0),
      _random_bools() {
  precompute_bools();
}

void Random::reinit(const int seed) {
  _generator = std::mt19937(seed + tbb::this_task_arena::current_thread_index());
  _next_random_bool = 0;
  precompute_bools();
}

void Random::precompute_bools() {
  std::uniform_int_distribution<int> _dist(0, 1);
  for (std::size_t i = 0; i < kPrecomputedBools; ++i) {
    _random_bools[i] = static_cast<bool>(_dist(_generator));
  }
}
} // namespace kaminpar
