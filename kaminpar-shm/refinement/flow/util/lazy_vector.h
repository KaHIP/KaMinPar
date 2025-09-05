#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace kaminpar::shm {

template <typename T> class LazyVector {
public:
  using Factory = std::function<T()>;

  LazyVector(const std::size_t capacity) : _factory([] { return T(); }), _data(capacity) {}

  LazyVector(Factory factory, const std::size_t capacity)
      : _factory(std::move(factory)),
        _data(capacity) {}

  T &operator[](const std::size_t idx) {
    KASSERT(idx < _data.size());

    if (!_data[idx]) {
      _data[idx] = std::make_unique<T>(_factory());
    }

    return *_data[idx];
  }

  template <typename Callback> void for_each(Callback &&callback) {
    for (std::size_t i = 0; i < _data.size(); ++i) {
      if (_data[i]) {
        callback(*_data[i]);
      }
    }
  }

private:
  Factory _factory;
  std::vector<std::unique_ptr<T>> _data;
};

} // namespace kaminpar::shm
