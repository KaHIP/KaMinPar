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

  LazyVector(Factory factory) : _factory(std::move(factory)) {}

  LazyVector(Factory factory, const std::size_t size) : _factory(std::move(factory)), _data(size) {}

  T &operator[](const std::size_t idx) {
    if (idx >= _data.size()) {
      _data.resize(idx + 1);
    }

    if (!_data[idx]) {
      _data[idx] = std::make_unique<T>(_factory());
    }

    return *_data[idx];
  }

private:
  Factory _factory;
  std::vector<std::unique_ptr<T>> _data;
};

} // namespace kaminpar::shm
