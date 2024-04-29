/*******************************************************************************
 * @file   aligned_element.h
 * @author Daniel Seemaier
 * @date   20.12.2021
 * @brief  Wrapper that aligns values to a cache line.
 ******************************************************************************/
#pragma once

#include <cstdlib>

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

  decltype(auto) operator[](std::size_t pos) {
    return vec[pos];
  }

  decltype(auto) operator[](std::size_t pos) const {
    return vec[pos];
  }

  decltype(auto) begin() noexcept {
    return vec.begin();
  }

  decltype(auto) begin() const noexcept {
    return vec.begin();
  }

  decltype(auto) end() noexcept {
    return vec.end();
  }

  decltype(auto) end() const noexcept {
    return vec.end();
  }

  void clear() noexcept {
    vec.clear();
  }

  void resize(std::size_t count) {
    vec.resize(count);
  }
};

} // namespace kaminpar::parallel
