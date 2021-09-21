/*******************************************************************************
 * @file:   math.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Math helper functions.
 ******************************************************************************/
#pragma once

#include <concepts>
#include <utility>
#include <vector>

namespace kaminpar::math {
//! With `UInt = uint32_t`, same as `static_cast<uint32_t>(std::log2(arg))`
template<std::unsigned_integral T>
T floor_log2(const T arg) {
  constexpr std::size_t arg_width{std::numeric_limits<T>::digits};

  auto log2{static_cast<T>(arg_width)};
  if constexpr (arg_width == std::numeric_limits<unsigned int>::digits) {
    log2 -= __builtin_clz(arg);
  } else {
    static_assert(arg_width == std::numeric_limits<unsigned long>::digits, "unsupported data type width");
    log2 -= __builtin_clzl(arg);
  }

  return log2 - 1;
}

//! With `UInt = uint32_t`, same as `static_cast<uint32_t>(std::ceil(std::log2(arg)))`
template<std::unsigned_integral T>
T ceil_log2(const T arg) {
  return floor_log2<T>(arg) + 1 - ((arg & (arg - 1)) == 0);
}

//! Checks whether `arg` is a power of 2.
template<std::integral T>
constexpr bool is_power_of_2(const T arg) {
  return arg && ((arg & (arg - 1)) == 0);
}

template<typename E>
double percentile(const std::vector<E> &sorted_sequence, const double percentile) {
  ASSERT([&] {
    for (std::size_t i = 1; i < sorted_sequence.size(); ++i) {
      if (sorted_sequence[i - 1] > sorted_sequence[i]) { return false; }
    }
    return true;
  });
  ASSERT(0 <= percentile && percentile <= 1);

  return sorted_sequence[std::ceil(percentile * sorted_sequence.size()) - 1];
}

template<std::integral T>
auto split_integral(const T value, const double ratio = 0.5) {
  return std::pair{static_cast<T>(std::ceil(value * ratio)), static_cast<T>(std::floor(value * (1.0 - ratio)))};
}

auto round_down_to_power_of_2(const std::unsigned_integral auto value) { return 1 << floor_log2(value); }

auto round_up_to_power_of_2(const std::unsigned_integral auto value) { return 1 << ceil_log2(value); }
} // namespace kaminpar::math