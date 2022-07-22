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
template <typename Container, typename T>
T accumulate(const Container& r, T initial) {
    using size_t = typename Container::size_type;

    class body {
        const Container& _r;

    public:
        T _ans{};

        void operator()(const tbb::blocked_range<size_t>& indices) {
            const Container& r   = _r;
            auto             ans = _ans;
            auto             end = indices.end();
            for (auto i = indices.begin(); i != end; ++i) {
                ans += r[i];
            }
            _ans = ans;
        }

        void join(const body& y) {
            _ans += y._ans;
        }

        body(body& x, tbb::split) : _r{x._r} {}
        body(const Container& r) : _r{r} {}
    };

    body b{r};
    tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
    return initial + b._ans;
}

template <typename InputIt, typename T>
T accumulate(InputIt begin, InputIt end, T initial) {
    using size_t = typename std::iterator_traits<InputIt>::difference_type;

    class body {
        const InputIt _begin;

    public:
        T _ans{};

        void operator()(const tbb::blocked_range<size_t>& indices) {
            const InputIt begin = _begin;
            auto          ans   = _ans;
            auto          end   = indices.end();
            for (auto i = indices.begin(); i != end; ++i) {
                ans += *(begin + i);
            }
            _ans = ans;
        }

        void join(const body& y) {
            _ans += y._ans;
        }

        body(body& x, tbb::split) : _begin{x._begin} {}
        body(const InputIt begin) : _begin{begin} {}
    };

    body b{begin};
    tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b);
    return initial + b._ans;
}

template <typename Container>
typename Container::value_type max_element(const Container& r) {
    using size_t  = typename Container::size_type;
    using value_t = typename Container::value_type;

    class body {
        const Container& _r;

    public:
        value_t _ans{std::numeric_limits<value_t>::min()};

        void operator()(const tbb::blocked_range<size_t>& indices) {
            const Container& r   = _r;
            auto             ans = _ans;
            auto             end = indices.end();
            for (auto i = indices.begin(); i != end; ++i) {
                ans = std::max<value_t>(ans, r[i]);
            }
            _ans = ans;
        }

        void join(const body& y) {
            _ans = std::max(_ans, y._ans);
        }

        body(body& x, tbb::split) : _r{x._r} {}
        body(const Container& r) : _r{r} {}
    };

    body b{r};
    tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), r.size()), b);
    return b._ans;
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type max_element(InputIt begin, InputIt end) {
    using size_t  = typename std::iterator_traits<InputIt>::difference_type;
    using value_t = typename std::iterator_traits<InputIt>::value_type;

    class body {
        InputIt _begin;

    public:
        value_t _ans{std::numeric_limits<value_t>::min()};

        void operator()(const tbb::blocked_range<size_t>& indices) {
            InputIt begin = _begin;
            auto    ans   = _ans;
            auto    end   = indices.end();

            for (auto i = indices.begin(); i != end; ++i) {
                ans = std::max<value_t>(ans, *(begin + i));
            }
            _ans = ans;
        }

        void join(const body& y) {
            _ans = std::max(_ans, y._ans);
        }

        body(body& x, tbb::split) : _begin{x._begin} {}
        body(InputIt begin) : _begin{begin} {}
    };

    body b{begin};
    tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), std::distance(begin, end)), b);
    return b._ans;
}

template <typename InputIterator, typename OutputIterator>
void prefix_sum(InputIterator first, InputIterator last, OutputIterator result) {
    using size_t = std::size_t;                    // typename InputIterator::difference_type;
    using Value  = std::decay_t<decltype(*first)>; // typename InputIterator::value_type;

    const size_t n = std::distance(first, last);
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, n), Value(),
        [first, result](const tbb::blocked_range<size_t>& r, Value sum, bool is_final_scan) {
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
} // namespace kaminpar::parallel
