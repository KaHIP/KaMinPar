/*******************************************************************************
 * @file   aligned_element.h
 * @author Daniel Seemaier
 * @date   20.12.2021
 * @brief  Wrapper that aligns values to a cache line.
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <utility>

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

template <typename Vector> struct alignas(64) AlignedVec {
  using value_type = typename Vector::value_type;
  using size_type = typename Vector::size_type;

  Vector vec;

  AlignedVec() noexcept : vec() {}
  AlignedVec(Vector vec) noexcept : vec(std::move(vec)) {}

  decltype(auto) operator[](size_type pos) {
    return vec[pos];
  }

  decltype(auto) operator[](size_type pos) const {
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

  decltype(auto) size() const {
    return vec.size();
  }

  void clear() noexcept {
    vec.clear();
  }

  void resize(size_type count) {
    vec.resize(count);
  }

  void resize(size_type count, const value_type &value) {
    vec.resize(count, value);
  }
};

} // namespace kaminpar::parallel
