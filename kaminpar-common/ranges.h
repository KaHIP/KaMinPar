/*******************************************************************************
 * C++-17 implementation of `std::ranges::iota` etc.
 *
 * @file:   ranges.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <iterator>
#include <type_traits>

namespace kaminpar {
template <typename Int> class IotaRange {
public:
  class iterator {
    friend IotaRange;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Int;
    using difference_type = std::make_signed_t<Int>;
    using pointer = Int *;
    using reference = Int &;

    explicit iterator(const Int value) : _value(value) {}

    Int operator*() const {
      return _value;
    }

    iterator &operator++() {
      ++_value;
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const iterator &other) const {
      return _value == other._value;
    }
    bool operator!=(const iterator &other) const {
      return _value != other._value;
    }

  private:
    Int _value;
  };

  IotaRange(const Int begin, const Int end) : _begin(begin), _end(end) {}

  iterator begin() const {
    return _begin;
  }
  iterator end() const {
    return _end;
  }

private:
  iterator _begin;
  iterator _end;
};

template <typename Int, typename Function> class TransformedIotaRange {
public:
  class iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::result_of_t<Function(Int)>;
    using difference_type = std::make_signed_t<Int>;
    using pointer = value_type *;
    using reference = value_type &;

    explicit iterator(const Int value, const Function transformer)
        : _value(value),
          _transformer(transformer) {}

    value_type operator*() const {
      return _transformer(_value);
    }

    iterator &operator++() {
      ++_value;
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    iterator operator+(const int step) const {
      auto tmp = *this;
      tmp._value += step;
      return tmp;
    }

    bool operator==(const iterator &other) const {
      return _value == other._value;
    }
    bool operator!=(const iterator &other) const {
      return _value != other._value;
    }

  private:
    Int _value;
    Function _transformer;
  };

  TransformedIotaRange(const Int begin, const Int end, const Function transformer)
      : _begin(begin, transformer),
        _end(end, transformer) {}

  iterator begin() const {
    return _begin;
  }
  iterator end() const {
    return _end;
  }

private:
  iterator _begin;
  iterator _end;
};

template <typename Iterator, typename Function> class TransformedRange {
public:
  class iterator {
  public:
    using iterator_category = typename Iterator::iterator_category;
    using value_type = std::result_of_t<Function(typename Iterator::value_type)>;
    using difference_type = typename Iterator::difference_type;
    using pointer = value_type *;
    using reference = value_type &;

    explicit iterator(Iterator iter, Function transformer)
        : _iter(iter),
          _transformer(transformer) {}

    value_type operator*() const {
      return _transformer(*_iter);
    }

    iterator &operator++() {
      ++_iter;
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      _iter++;
      return tmp;
    }

    bool operator==(const iterator &other) const {
      return _iter == other._iter;
    }
    bool operator!=(const iterator &other) const {
      return _iter != other._iter;
    }

  private:
    Iterator _iter;
    Function _transformer;
  };

  TransformedRange(Iterator begin, Iterator end, Function transformer)
      : _begin(begin, transformer),
        _end(end, transformer) {}

  iterator begin() {
    return _begin;
  }
  iterator end() {
    return _end;
  }

private:
  iterator _begin;
  iterator _end;
};
} // namespace kaminpar
