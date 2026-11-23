/*******************************************************************************
 * Thread-local singleton for PRNG.
 *
 * @file:   random.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/random.h"

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

int Random::get_seed() {
  return _seed;
}

void Random::reseed(const int seed) {
  _seed = seed;
  for (auto &instance : _instances) {
    instance->reinit(seed);
  }
}

void Random::reinit(const int seed) {
  _local_seed = seed;
  reset();
}

void Random::reset() {
  const std::uint64_t seed = static_cast<std::uint64_t>(
      _local_seed + (_thread_independent_seeding ? 0 : tbb::this_task_arena::current_thread_index())
  );
  _generator.seed(seed);
}

} // namespace kaminpar
