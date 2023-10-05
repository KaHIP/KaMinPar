/*******************************************************************************
 * @file:   accumulate.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Helper classes and functions for parallel programming.
 ******************************************************************************/
#pragma once

#include <iterator>
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>

namespace kaminpar::parallel {
template <typename Container>
typename Container::value_type
accumulate(const Container &r, typename Container::value_type initial) {
  using size_t = typename Container::size_type;
  using value_t = typename Container::value_type;

  class body {
    const Container &_r;

  public:
    value_t _ans{};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const Container &r = _r;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) {
        ans += r[i];
      }
      _ans = ans;
    }

    void join(const body &y) {
      _ans += y._ans;
    }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Container &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
  return initial + b._ans;
}

template <
    typename InputIt,
    typename UnaryOperation,
    typename ValueType =
        std::result_of_t<UnaryOperation(typename std::iterator_traits<InputIt>::value_type)>>
ValueType accumulate(InputIt begin, InputIt end, ValueType initial, UnaryOperation op) {
  using size_t = typename std::iterator_traits<InputIt>::difference_type;
  using value_t = ValueType;

  class body {
    const InputIt _begin;
    UnaryOperation _op;

  public:
    value_t _ans{};

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const InputIt begin = _begin;
      auto ans = _ans;
      auto end = indices.end();
      for (auto i = indices.begin(); i != end; ++i) {
        ans += _op(*(begin + i));
      }
      _ans = ans;
    }

    void join(const body &y) {
      _ans += y._ans;
    }

    body(body &x, tbb::split) : _begin{x._begin}, _op{x._op} {}
    body(const InputIt begin, UnaryOperation op) : _begin{begin}, _op{op} {}
  };

  body b{begin, op};
  tbb::parallel_reduce(
      tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b
  );
  return initial + b._ans;
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type
accumulate(InputIt begin, InputIt end, typename std::iterator_traits<InputIt>::value_type initial) {
  return ::kaminpar::parallel::accumulate(begin, end, initial, [](const auto &v) { return v; });
}

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

    void join(const body &y) {
      _ans = std::max(_ans, y._ans);
    }

    body(body &x, tbb::split) : _r{x._r} {}
    body(const Container &r) : _r{r} {}
  };

  body b{r};
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
  return b._ans;
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type max_element(InputIt begin, InputIt end) {
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

    void join(const body &y) {
      _ans = std::max(_ans, y._ans);
    }

    body(body &x, tbb::split) : _begin{x._begin} {}
    body(InputIt begin) : _begin{begin} {}
  };

  body b{begin};
  tbb::parallel_reduce(
      tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b
  );
  return b._ans;
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type max_difference(InputIt begin, InputIt end) {
  using size_t = typename std::iterator_traits<InputIt>::difference_type;
  using value_t = typename std::iterator_traits<InputIt>::value_type;

  const std::size_t size = std::distance(begin, end);

  // Catch special cases: zero or one element
  if (size == 0) {
    return std::numeric_limits<value_t>::min();
  } else if (size == 1) {
    return 0;
  }

  class body {
    InputIt _begin;

  public:
    value_t _ans = std::numeric_limits<value_t>::min();

    void operator()(const tbb::blocked_range<size_t> &indices) {
      const InputIt begin = _begin;
      const auto end = indices.end();

      auto ans = _ans;
      for (auto i = indices.begin(); i != end; ++i) {
        ans = std::max<value_t>(ans, *(begin + i + 1) - *(begin + i));
      }
      _ans = ans;
    }

    void join(const body &y) {
      _ans = std::max(_ans, y._ans);
    }

    body(body &x, tbb::split) : _begin{x._begin} {}
    body(InputIt begin) : _begin{begin} {}
  };

  body b(begin);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size - 1), b);
  return b._ans;
}

template <typename Container>
typename Container::value_type max_difference(const Container &container) {
  return max_difference(std::begin(container), std::end(container));
}

template <typename InputIterator, typename OutputIterator>
void prefix_sum(InputIterator first, InputIterator last, OutputIterator result) {
  using size_t = std::size_t;                   // typename InputIterator::difference_type;
  using Value = std::decay_t<decltype(*first)>; // typename InputIterator::value_type;

  const size_t n = std::distance(first, last);
  tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n),
      Value(),
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
      [](Value left, Value right) { return left + right; }
  );
}
} // namespace kaminpar::parallel
