/*******************************************************************************
 * @file:   math.h
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 * @brief:  Simple math utility functions
 ******************************************************************************/
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <kassert/kassert.hpp>

namespace kaminpar::math {
template <typename Int>
bool is_square(const Int value) {
    const Int sqrt = std::sqrt(value);
    return sqrt * sqrt == value;
}

//! Checks whether `arg` is a power of 2.
template <typename T>
constexpr bool is_power_of_2(const T arg) {
    return arg && ((arg & (arg - 1)) == 0);
}

//! With `UInt = uint32_t`, same as `static_cast<uint32_t>(std::log2(arg))`
template <typename T>
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

template <typename T>
T floor2(const T arg) {
    1 << floor_log2(arg);
}

//! With `UInt = uint32_t`, same as `static_cast<uint32_t>(std::ceil(std::log2(arg)))`
template <typename T>
T ceil_log2(const T arg) {
    return floor_log2<T>(arg) + 1 - ((arg & (arg - 1)) == 0);
}

template <typename T>
T ceil2(const T arg) {
    return 1 << ceil_log2(arg);
}

template <typename E>
double percentile(const std::vector<E>& sorted_sequence, const double percentile) {
    KASSERT([&] {
        for (std::size_t i = 1; i < sorted_sequence.size(); ++i) {
            if (sorted_sequence[i - 1] > sorted_sequence[i]) {
                return false;
            }
        }
        return true;
    }());
    KASSERT((0 <= percentile && percentile <= 1));

    return sorted_sequence[std::ceil(percentile * sorted_sequence.size()) - 1];
}

template <typename T>
auto split_integral(const T value, const double ratio = 0.5) {
    return std::pair{static_cast<T>(std::ceil(value * ratio)), static_cast<T>(std::floor(value * (1.0 - ratio)))};
}
} // namespace kaminpar::math
