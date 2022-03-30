/*******************************************************************************
 * @file:   accumulate.h
 *
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Helper classes and functions for parallel programming.
 ******************************************************************************/
#pragma once

#include <iterator>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace kaminpar::parallel {
template <typename Container, typename T> T accumulate(const Container &r, T initial) {
  using size_t = typename Container::size_type;

  class body {
    const Container &_r;

  public:
    T _ans{};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const Container &r = _r;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) {
        ans += r[i];
      }
      _ans = ans;
    }

    void join(const body &y) { _ans += y._ans; }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Container &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
  return initial + b._ans;
}

template <typename InputIt, typename T> T accumulate(InputIt begin, InputIt end, T initial) {
  using size_t = typename std::iterator_traits<InputIt>::difference_type;

  class body {
    const InputIt _begin;

  public:
    T _ans{};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const InputIt begin = _begin;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) {
        ans += *(_begin + i);
      }
      _ans = ans;
    }

    void join(const body &y) { _ans += y._ans; }

    body(body &x, tbb::split) : _begin{x._begin} {}
    body(const InputIt begin) : _begin{begin} {}
  };

  body b{begin};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b);
  return initial + b._ans;
}
} // namespace kaminpar::parallel
