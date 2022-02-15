/*******************************************************************************
 * @file:   random.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper class for randomization.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/math.h"
#include "kaminpar/utility/strings.h"

#include <algorithm>
#include <random>
#include <tbb/task_arena.h>

namespace kaminpar {
class Randomize {
  static constexpr std::size_t kPrecomputedBools = 1024;
  static_assert(math::is_power_of_2(kPrecomputedBools), "not a power of 2");

public:
  Randomize()
      : _generator(Randomize::seed + tbb::this_task_arena::current_thread_index()),
        _bool_dist(0, 1),
        _next_random_bool(0),
        _random_bools{} {
    precompute_bools();
  }

  static Randomize &instance();

  Randomize(const Randomize &) = delete;
  Randomize &operator=(const Randomize &) = delete;

  Randomize(Randomize &&) noexcept = default; // might be expensive
  Randomize &operator=(Randomize &&) = delete;

  using generator_type = std::mt19937;

  std::size_t random_index(const std::size_t inclusive_lower_bound, const std::size_t exclusive_upper_bound) {
    ASSERT(exclusive_upper_bound > 0);
    return std::uniform_int_distribution<std::size_t>(inclusive_lower_bound, exclusive_upper_bound - 1)(_generator);
  }

  NodeID random_node(const Graph &graph) {
    return static_cast<NodeID>(random_index(0, static_cast<std::size_t>(graph.n())));
  }

  bool random_bool() { return _random_bools[_next_random_bool++ % kPrecomputedBools]; }
  bool random_bool(const double prob) { return std::uniform_real_distribution<>(0, 1)(_generator) <= prob; }

  template<typename Container>
  void shuffle(Container &&vec) { std::shuffle(vec.begin(), vec.end(), _generator); }

  template<typename Iterator>
  void shuffle(Iterator begin, Iterator end) { std::shuffle(begin, end, _generator); }

  [[nodiscard]] auto &generator() { return _generator; }

  static int seed;

private:
  void precompute_bools() {
    std::uniform_int_distribution<int> _dist(0, 1);
    for (std::size_t i = 0; i < kPrecomputedBools; ++i) { _random_bools[i] = static_cast<bool>(_dist(_generator)); }
  }

  std::mt19937 _generator;
  std::uniform_int_distribution<int> _bool_dist;
  std::size_t _next_random_bool;
  std::array<bool, kPrecomputedBools> _random_bools;
};

template<typename ValueType, std::size_t size, std::size_t count>
class RandomPermutations {
public:
  RandomPermutations() : _rand{Randomize::instance()} { init_permutations(); }

  RandomPermutations(const RandomPermutations &) = delete;
  RandomPermutations &operator=(const RandomPermutations &) = delete;
  RandomPermutations(RandomPermutations &&) = delete;
  RandomPermutations &operator=(RandomPermutations &&) = delete;

  const std::array<ValueType, size> &get(Randomize &rand) {
    return _permutations[rand.random_index(0, _permutations.size())];
  }

  const std::array<ValueType, size> &get() { return get(_rand); }

private:
  void init_permutations() {
    for (auto &permutation : _permutations) {
      std::iota(permutation.begin(), permutation.end(), 0);
      _rand.shuffle(permutation);
    }
  }

  Randomize &_rand;
  std::array<std::array<ValueType, size>, count> _permutations{};
};
} // namespace kaminpar
