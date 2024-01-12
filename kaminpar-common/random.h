/*******************************************************************************
 * Thread-local singleton for PRNG.
 *
 * @file:   random.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "kaminpar-common/math.h"

namespace kaminpar {
class Random {
  static constexpr std::size_t kPrecomputedBools = 1024;
  static_assert(math::is_power_of_2(kPrecomputedBools), "not a power of 2");

public:
  Random();

  static Random &instance();
  static void reseed(int seed);
  static int get_seed();

  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;

  Random(Random &&) noexcept = default; // might be expensive
  Random &operator=(Random &&) = delete;

  using generator_type = std::mt19937;

  void reinit(int seed);

  std::size_t
  random_index(const std::size_t inclusive_lower_bound, const std::size_t exclusive_upper_bound) {
    return std::uniform_int_distribution<std::size_t>(
        inclusive_lower_bound, exclusive_upper_bound - 1
    )(_generator);
  }

  bool random_bool() {
    return _random_bools[_next_random_bool++ % kPrecomputedBools];
  }
  bool random_bool(const double prob) {
    return _real_dist(_generator) <= prob;
  }

  template <typename Container> void shuffle(Container &&vec) {
    std::shuffle(vec.begin(), vec.end(), _generator);
  }

  template <typename Iterator> void shuffle(Iterator begin, Iterator end) {
    std::shuffle(begin, end, _generator);
  }

  [[nodiscard]] auto &generator() {
    return _generator;
  }

private:
  static int _seed;
  static std::mutex _create_mutex;
  static std::vector<std::unique_ptr<Random>> _instances;
  static Random &create_instance();

  void precompute_bools();

  std::mt19937 _generator;
  std::uniform_int_distribution<int> _bool_dist;
  std::uniform_real_distribution<> _real_dist;
  std::size_t _next_random_bool;
  std::array<bool, kPrecomputedBools> _random_bools;
};

template <typename ValueType, std::size_t size, std::size_t count> class RandomPermutations {
public:
  RandomPermutations(Random &rand) : _rand(rand) {
    init_permutations();
  }

  RandomPermutations() : _rand(Random::instance()) {
    init_permutations();
  }

  RandomPermutations(const RandomPermutations &) = delete;
  RandomPermutations &operator=(const RandomPermutations &) = delete;

  RandomPermutations(RandomPermutations &&) = delete;
  RandomPermutations &operator=(RandomPermutations &&) = delete;

  const std::array<ValueType, size> &get(Random &rand) {
    return _permutations[rand.random_index(0, _permutations.size())];
  }

  const std::array<ValueType, size> &get() {
    return get(_rand);
  }

private:
  void init_permutations() {
    for (auto &permutation : _permutations) {
      std::iota(permutation.begin(), permutation.end(), 0);
      _rand.shuffle(permutation);
    }
  }

  Random &_rand;
  std::array<std::array<ValueType, size>, count> _permutations{};
};
} // namespace kaminpar
