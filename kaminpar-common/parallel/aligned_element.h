/*******************************************************************************
 * @file   aligned_element.h
 * @author Daniel Seemaier
 * @date   20.12.2021
 * @brief  Wrapper that aligns values to a cache line.
 ******************************************************************************/
#pragma once

#include <type_traits>

namespace kaminpar::parallel {

template <typename Value> struct alignas(64) Aligned {
  Value value;

  Aligned() : value() {}
  Aligned(Value value) : value(value) {}

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

template <typename Vector> struct alignas(64) AlignedVec {
  Vector vec;

  AlignedVec() : vec() {}
  AlignedVec(Vector vec) : vec(vec) {}

  auto begin() noexcept {
    return vec.begin();
  }

  auto begin() const noexcept {
    return vec.begin();
  }

  auto end() noexcept {
    return vec.end();
  }

  auto end() const noexcept {
    return vec.end();
  }

  void clear() noexcept {
    vec.clear();
  }
};

} // namespace kaminpar::parallel
