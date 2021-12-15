#pragma once

#include "kaminpar/definitions.h"

#include <atomic>
#include <iterator>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/scalable_allocator.h>

#ifndef __linux__
inline int sched_getcpu() { return 0; }
#endif

namespace kaminpar::parallel {
// https://github.com/kahypar/mt-kahypar/blob/master/mt-kahypar/parallel/atomic_wrapper.h
template<typename T>
class IntegralAtomicWrapper {
public:
  IntegralAtomicWrapper(const T value = T()) : _value(value) {}

  IntegralAtomicWrapper(const IntegralAtomicWrapper &other) : _value(other._value.load()) {}

  IntegralAtomicWrapper &operator=(const IntegralAtomicWrapper &other) {
    _value = other._value.load();
    return *this;
  }

  IntegralAtomicWrapper(IntegralAtomicWrapper &&other) noexcept : _value(other._value.load()) {}

  IntegralAtomicWrapper &operator=(IntegralAtomicWrapper &&other) noexcept {
    _value = other._value.load();
    return *this;
  }

  T operator=(T desired) noexcept {
    _value.store(desired, std::memory_order_relaxed);
    return _value;
  }

  void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept { _value.store(desired, order); }

  T load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return _value.load(order); }

  operator T() const noexcept { return _value.load(std::memory_order_relaxed); }

  T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.exchange(desired, order);
  }

  bool compare_exchange_weak(T &expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    return _value.compare_exchange_weak(expected, desired, order);
  }

  bool compare_exchange_strong(T &expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
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

  T operator++() noexcept { return ++_value; }

  T operator++(int) noexcept { return _value++; }

  T operator--() noexcept { return --_value; }

  T operator--(int) noexcept { return _value++; }

  T operator+=(T arg) noexcept { return _value.operator+=(arg); }

  T operator-=(T arg) noexcept { return _value.operator-=(arg); }

  T operator&=(T arg) noexcept { return _value.operator&=(arg); }

  T operator|=(T arg) noexcept { return _value.operator|=(arg); }

  T operator^=(T arg) noexcept { return _value.operator^=(arg); }

private:
  std::atomic<T> _value;
};

template<typename InputIterator, typename OutputIterator>
void prefix_sum(InputIterator first, InputIterator last, OutputIterator result) {
  using size_t = std::size_t;                   //typename InputIterator::difference_type;
  using Value = std::decay_t<decltype(*first)>; //typename InputIterator::value_type;

  const size_t n = std::distance(first, last);
  tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n), Value(),
      [first, result](const tbb::blocked_range<size_t> &r, Value sum, bool is_final_scan) {
        Value temp = sum;
        for (auto i = r.begin(); i < r.end(); ++i) {
          temp += *(first + i);
          if (is_final_scan) { *(result + i) = temp; }
        }
        return temp;
      },
      [](Value left, Value right) { return left + right; });
}

template<typename T>
struct tbb_deleter {
  void operator()(T *p) { scalable_free(p); }
};

template<typename T>
using tbb_unique_ptr = std::unique_ptr<T, tbb_deleter<T>>;

template<typename T>
tbb_unique_ptr<T> make_unique(const std::size_t size) {
  T *ptr = static_cast<T *>(scalable_malloc(sizeof(T) * size));
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}

template<typename T, typename... Args>
tbb_unique_ptr<T> make_unique(Args &&...args) {
  void *memory = static_cast<T *>(scalable_malloc(sizeof(T)));
  T *ptr = new (memory) T(std::forward<Args...>(args)...);
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}

template<typename Range>
typename Range::value_type accumulate(const Range &r) {
  using r_size_t = typename Range::size_type;
  using value_t = typename Range::value_type;

  class body {
    const Range &_r;

  public:
    value_t _ans{};

    void operator()(const tbb::blocked_range<r_size_t> &indices) {
      const Range &r = _r;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) { ans += r[i]; }
      _ans = ans;
    }

    void join(const body &y) { _ans += y._ans; }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Range &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range(static_cast<r_size_t>(0), r.size()), b);
  return b._ans;
}

template<typename Range>
typename Range::value_type max_element(const Range &r) {
  using r_size_t = typename Range::size_type;
  using value_t = typename Range::value_type;

  class body {
    const Range &_r;

  public:
    value_t _ans{std::numeric_limits<value_t>::min()};

    void operator()(const tbb::blocked_range<r_size_t> &indices) {
      const Range &r = _r;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) { ans = std::max<value_t>(ans, r[i]); }
      _ans = ans;
    }

    void join(const body &y) { _ans = std::max(_ans, y._ans); }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Range &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range(static_cast<r_size_t>(0), r.size()), b);
  return b._ans;
}
} // namespace kaminpar::parallel