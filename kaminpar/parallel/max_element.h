/*******************************************************************************
 * @file:   max_element.h
 *
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Finds the maximum element of a range in parallel.
 ******************************************************************************/
#pragma once

#include <iterator>
#include <limits>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace kaminpar::parallel {
template <typename Container> typename Container::value_type max_element(const Container &r) {
  using size_t = typename Container::size_type;
  using value_t = typename Container::value_type;

  class body {
    const Container &_r;

  public:
    value_t _ans{std::numeric_limits<value_t>::min()};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const Container &r = _r;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) {
        ans = std::max<value_t>(ans, r[i]);
      }
      _ans = ans;
    }

    void join(const body &y) { _ans = std::max(_ans, y._ans); }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Container &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
  return b._ans;
}

template <typename InputIt> typename std::iterator_traits<InputIt>::value_type max_element(InputIt begin, InputIt end) {
  using size_t = typename std::iterator_traits<InputIt>::difference_type;
  using value_t = typename std::iterator_traits<InputIt>::value_type;

  class body {
    InputIt _begin;

  public:
    value_t _ans{std::numeric_limits<value_t>::min()};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      InputIt begin = _begin;
      auto ans = _ans;
      auto end = indices.end();

      for (auto i = indices.begin(); i != end; ++i) {
        ans = std::max<value_t>(ans, *(begin + i));
      }
      _ans = ans;
    }

    void join(const body &y) { _ans = std::max(_ans, y._ans); }

    body(body &x, tbb::split) : _begin{x._begin} {}
    body(InputIt begin) : _begin{begin} {}
  };

  body b{begin};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b);
  return b._ans;
}
} // namespace kaminpar
