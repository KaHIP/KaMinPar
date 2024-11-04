/*******************************************************************************
 * Value wrapper that aligns to cache lines.
 *
 * @file   aligned_element.h
 * @author Daniel Seemaier
 * @date   20.12.2021
 ******************************************************************************/
#pragma once

#include <cstdlib>

namespace kaminpar::parallel {

template <typename Value> struct alignas(64) Aligned {
  Value value;

  Aligned() noexcept : value() {}
  Aligned(Value value) noexcept : value(value) {}

  Aligned<Value> &operator++() {
    ++value;
    return *this;
  }

  Aligned<Value> &operator--() {
    --value;
    return *this;
  }

  bool operator==(const Value &other) const {
    return value == other;
  }

  bool operator!=(const Value &other) const {
    return value != other;
  }
};

} // namespace kaminpar::parallel
