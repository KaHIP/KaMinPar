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
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/math.h"

namespace kaminpar {

namespace random {
constexpr struct thread_independent_seeding_t {
} thread_independent_seeding;
} // namespace random

class Xoshiro256PlusPlus {
public:
  using result_type = std::uint64_t;

  explicit Xoshiro256PlusPlus(const std::uint64_t seed = 0) {
    this->seed(seed);
  }

  void seed(std::uint64_t x) {
    for (std::uint64_t &s : _s) {
      x += 0x9e3779b97f4a7c15ULL;
      std::uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      s = z ^ (z >> 31);
    }
  }

  [[nodiscard]] KAMINPAR_INLINE result_type operator()() {
    const std::uint64_t result = std::rotl(_s[0] + _s[3], 23) + _s[0];
    const std::uint64_t t = _s[1] << 17;
    _s[2] ^= _s[0];
    _s[3] ^= _s[1];
    _s[1] ^= _s[2];
    _s[0] ^= _s[3];
    _s[2] ^= t;
    _s[3] = std::rotl(_s[3], 45);
    return result;
  }

  [[nodiscard]] static constexpr result_type min() {
    return 0;
  }

  [[nodiscard]] static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

  bool operator==(const Xoshiro256PlusPlus &) const = default;

private:
  std::array<std::uint64_t, 4> _s{};
};

class Random {
public:
  [[nodiscard]] static Random &instance();

  [[nodiscard]] static int get_seed();

  static void reseed(int seed);

public:
  template <typename... Tags> Random(Tags...) {
    static_assert(
        tag::are_all_contained<Tags...>(random::thread_independent_seeding), "invalid tags"
    );

    if constexpr (contains_tag_v<random::thread_independent_seeding_t, Tags...>) {
      _thread_independent_seeding = true;
    } else {
      _thread_independent_seeding = false;
    }

    _local_seed = _seed;
    reset();
  }

  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;

  Random(Random &&) noexcept = default;
  Random &operator=(Random &&) = delete;

  void reinit(int seed);

  void reset();

  KAMINPAR_INLINE std::size_t
  random_index(const std::size_t inclusive_lower_bound, const std::size_t exclusive_upper_bound) {
    const std::size_t range = exclusive_upper_bound - inclusive_lower_bound;
    return inclusive_lower_bound +
           static_cast<std::size_t>((static_cast<__uint128_t>(_generator()) * range) >> 64);
  }

  KAMINPAR_INLINE bool random_bool() {
    return (_generator() >> 63) != 0;
  }

  KAMINPAR_INLINE bool random_bool(const double prob) {
    return random_double() <= prob;
  }

  KAMINPAR_INLINE double random_double() {
    return static_cast<double>(_generator() >> 11) * (1.0 / 9007199254740992.0);
  }

  template <typename Container> void shuffle(Container &&vec) {
    shuffle(vec.begin(), vec.end());
  }

  template <typename Iterator> void shuffle(Iterator begin, Iterator end) {
    const auto n = static_cast<std::size_t>(end - begin);
    for (std::size_t i = n; i > 1; --i) {
      const std::size_t j =
          static_cast<std::size_t>((static_cast<__uint128_t>(_generator()) * i) >> 64);
      std::iter_swap(begin + (i - 1), begin + j);
    }
  }

  [[nodiscard]] auto &generator() {
    return _generator;
  }

private:
  static Random &create_instance();

  static int _seed;

  static std::mutex _create_mutex;
  static std::vector<std::unique_ptr<Random>> _instances;

  int _local_seed;
  bool _thread_independent_seeding;

  Xoshiro256PlusPlus _generator;
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
