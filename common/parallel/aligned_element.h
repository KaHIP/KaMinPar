/*******************************************************************************
 * @file   aligned_element.h
 * @author Daniel Seemaier
 * @date   20.12.2021
 * @brief  Wrapper that aligns values to a cache line.
 ******************************************************************************/
#pragma once

#include <type_traits>

namespace kaminpar::parallel {
template <typename ValueT>
struct alignas(64) Aligned {
    ValueT value;

    template <std::enable_if_t<std::is_integral_v<ValueT>, int> = 0>
    Aligned<ValueT>& operator++() {
        ++value;
        return *this;
    }

    template <std::enable_if_t<std::is_integral_v<ValueT>, int> = 0>
    Aligned<ValueT>& operator--() {
        --value;
        return *this;
    }

    bool operator==(const ValueT& other) const {
        return value == other;
    }

    bool operator!=(const ValueT& other) const {
        return value != other;
    }
};
} // namespace kaminpar::parallel
