/*******************************************************************************
 * @file:   atomic.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Wrapper for std::atomic with copy ctor.
 ******************************************************************************/
#pragma once

#include <atomic>

namespace kaminpar::parallel {
template <typename T> class Atomic {
public:
  Atomic() {
    _value.store(0, std::memory_order_relaxed);
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  Atomic(const T value) : _value(value) {}

  Atomic(const Atomic &other) : _value(other._value.load()) {}

  Atomic &operator=(const Atomic &other) {
    _value = other._value.load();
    return *this;
  }

  Atomic(Atomic &&other) noexcept : _value(other._value.load()) {}

  Atomic &operator=(Atomic &&other) noexcept {
    _value = other._value.load();
    return *this;
  }

  Atomic &operator=(T desired) noexcept {
    _value.store(desired, std::memory_order_relaxed);
    return *this;
  }

  void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    _value.store(desired, order);
  }

  T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    return _value.load(order);
  }

  // NOLINTNEXTLINE
  operator T() const noexcept {
    return _value.load(std::memory_order_relaxed);
  }

  T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.exchange(desired, order);
  }

  bool compare_exchange_weak(
      T &expected, T desired, std::memory_order order = std::memory_order_seq_cst
  ) noexcept {
    return _value.compare_exchange_weak(expected, desired, order);
  }

  bool compare_exchange_strong(
      T &expected, T desired, std::memory_order order = std::memory_order_seq_cst
  ) noexcept {
    return _value.compare_exchange_strong(expected, desired, order);
  }

  T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.fetch_add(arg, order);
  }

  T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.fetch_sub(arg, order);
  }

  T fetch_and(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.fetch_and(arg, order);
  }

  T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.fetch_or(arg, order);
  }

  T fetch_xor(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.fetch_xor(arg, order);
  }

  T operator++() noexcept {
    return ++_value;
  }

  T operator++(int) &noexcept {
    return _value++;
  } // NOLINT

  T operator--() noexcept {
    return --_value;
  }

  T operator--(int) &noexcept {
    return _value++;
  } // NOLINT

  T operator+=(T arg) noexcept {
    return _value.operator+=(arg);
  }

  T operator-=(T arg) noexcept {
    return _value.operator-=(arg);
  }

  T operator&=(T arg) noexcept {
    return _value.operator&=(arg);
  }

  T operator|=(T arg) noexcept {
    return _value.operator|=(arg);
  }

  T operator^=(T arg) noexcept {
    return _value.operator^=(arg);
  }

private:
  std::atomic<T> _value;
};
} // namespace kaminpar::parallel
