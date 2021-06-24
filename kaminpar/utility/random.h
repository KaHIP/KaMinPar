#pragma once

#include "datastructure/graph.h"
#include "definitions.h"
#include "utility/utility.h"

#include <algorithm>
#include <random>

namespace kaminpar {
class Randomize {
  static constexpr std::size_t kPrecomputedBools = 1024;
  static_assert(math::is_power_of_2(kPrecomputedBools), "not a power of 2");

public:
  Randomize() : _generator(Randomize::seed), _bool_dist(0, 1), _next_random_bool(0), _random_bools{} {
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

  bool random_bool() { return _random_bools[_next_random_bool++ % kPrecomputedBools]; }

  NodeID random_node(const Graph &graph) {
    return static_cast<NodeID>(random_index(0, static_cast<std::size_t>(graph.n())));
  }

  template<typename Container>
  requires requires(Container c) {
    c.begin();
    c.end();
  }
  void shuffle(Container &&vec) { std::shuffle(vec.begin(), vec.end(), _generator); }
  void shuffle(auto begin, auto end) { std::shuffle(begin, end, _generator); }

  [[nodiscard]] auto &generator() { return _generator; }

#ifdef TEST
  void mock_precomputed_bools(const bool value) { std::fill(_random_bools.begin(), _random_bools.end(), value); }

  void mock_precomputed_bools(std::array<bool, kPrecomputedBools> mocked_random_bools) {
    _random_bools = mocked_random_bools;
  }
#endif // TEST

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