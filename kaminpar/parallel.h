/*******************************************************************************
 * @file:   parallel.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper classes and functions for parallel programming.
 ******************************************************************************/
#pragma once

#include "kaminpar/definitions.h"

#include <atomic>
#include <iterator>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/scalable_allocator.h>

#ifndef __linux__
inline int sched_getcpu() { return 0; }
#endif

namespace kaminpar {
namespace parallel {
// https://github.com/kahypar/mt-kahypar/blob/master/mt-kahypar/parallel/atomic_wrapper.h
template<typename T>
class IntegralAtomicWrapper {
public:
  IntegralAtomicWrapper() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  IntegralAtomicWrapper(const T value) : _value(value) {}

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

  IntegralAtomicWrapper &operator=(T desired) noexcept {
    _value.store(desired, std::memory_order_relaxed);
    return *this;
  }

  void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept { _value.store(desired, order); }

  T load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return _value.load(order); }

  // NOLINTNEXTLINE
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

  T operator++(int) &noexcept { return _value++; } // NOLINT

  T operator--() noexcept { return --_value; }

  T operator--(int) &noexcept { return _value++; } // NOLINT

  T operator+=(T arg) noexcept { return _value.operator+=(arg); }

  T operator-=(T arg) noexcept { return _value.operator-=(arg); }

  T operator&=(T arg) noexcept { return _value.operator&=(arg); }

  T operator|=(T arg) noexcept { return _value.operator|=(arg); }

  T operator^=(T arg) noexcept { return _value.operator^=(arg); }

private:
  std::atomic<T> _value = 0;
};

template <typename InputIterator, typename OutputIterator>
void prefix_sum(InputIterator first, InputIterator last, OutputIterator result) {
  using size_t = std::size_t;                   // typename InputIterator::difference_type;
  using Value = std::decay_t<decltype(*first)>; // typename InputIterator::value_type;

  const size_t n = std::distance(first, last);
  tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n), Value(),
      [first, result](const tbb::blocked_range<size_t> &r, Value sum, bool is_final_scan) {
        Value temp = sum;
        for (auto i = r.begin(); i < r.end(); ++i) {
          temp += *(first + i);
          if (is_final_scan) {
            *(result + i) = temp;
          }
        }
        return temp;
      },
      [](Value left, Value right) { return left + right; });
}

template <typename T> struct tbb_deleter {
  void operator()(T *p) { scalable_free(p); }
};

template <typename T> using tbb_unique_ptr = std::unique_ptr<T, tbb_deleter<T>>;

template <typename T> tbb_unique_ptr<T> make_unique(const std::size_t size) {
  T *ptr = static_cast<T *>(scalable_malloc(sizeof(T) * size));
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}

template <typename T, typename... Args> tbb_unique_ptr<T> make_unique(Args &&...args) {
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
      for (auto i = indices.begin(); i != end; ++i) {
        ans += r[i];
      }
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
      for (auto i = indices.begin(); i != end; ++i) {
        ans = std::max<value_t>(ans, r[i]);
      }
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

template<typename Buffer, typename Lambda>
void container_for(const Buffer &buffer, Lambda &&lambda) {
  tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) { lambda(buffer[i]); });
}

/*!
 * @param buffers Vector of buffers of elements.
 * @param lambda Invoked on each element, in parallel.
 */
template<typename Buffer, typename Lambda>
void chunked_for(Buffer &buffers, Lambda &&lambda) {
  std::size_t total_size = 0;
  for (const auto &buffer : buffers) {
    total_size += buffer.size();
  }

  constexpr bool invocable_with_chunk_id = std::is_invocable_v<decltype(lambda), decltype(buffers[0][0]), int>;

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, total_size), [&](const auto r) {
    std::size_t cur = r.begin();
    std::size_t offset = 0;
    std::size_t current_buf = 0;
    std::size_t cur_size = buffers[current_buf].size();

    // find first buffer for our range
    while (offset + cur_size < cur) {
      offset += cur_size;
      ++current_buf;
      ASSERT(current_buf < buffers.size());
      cur_size = buffers[current_buf].size();
    }

    // iterate elements
    while (cur != r.end()) {
      while (cur - offset >= cur_size) {
        ASSERT(current_buf < buffers.size());
        offset += buffers[current_buf++].size();
        cur_size = buffers[current_buf].size();
      }
      ASSERT(current_buf < buffers.size());
      ASSERT(cur_size == buffers[current_buf].size());
      ASSERT(cur - offset < buffers[current_buf].size());
      if constexpr (invocable_with_chunk_id) {
        lambda(buffers[current_buf][cur - offset], current_buf);
      } else {
        lambda(buffers[current_buf][cur - offset]);
      }
      ++cur;
    }
  });
}

/*!
 * Iterate over from..to in parallel, where from..from+x are processed by CPU 0, from+x..from+2x by CPU 1 etc.
 * The number of CPUs is determined by the maximum concurrency of the current TBB task arena.
 *
 * @tparam Index
 * @tparam Lambda Must take a start and end index and the CPU id.
 * @param from
 * @param to
 * @param lambda Called once for each CPU (in parallel): first element, first invalid element, CPU id
 */
template <typename Index, typename Lambda> void deterministic_for(const Index from, const Index to, Lambda &&lambda) {
  static_assert(std::is_invocable_v<Lambda, Index, Index, int>);

  const Index n = to - from;
  const int p = std::min<int>(tbb::this_task_arena::max_concurrency(), n);

  tbb::parallel_for(static_cast<int>(0), p, [&](const int cpu) {
    const NodeID chunk = n / p;
    const NodeID rem = n % p;
    const NodeID cpu_from = cpu * chunk + std::min(cpu, static_cast<int>(rem));
    const NodeID cpu_to = cpu_from + ((cpu < static_cast<int>(rem)) ? chunk + 1 : chunk);

    lambda(from + cpu_from, from + cpu_to, cpu);
  });
}
} // namespace parallel

template <typename T> using Atomic = parallel::IntegralAtomicWrapper<T>;
} // namespace kaminpar
