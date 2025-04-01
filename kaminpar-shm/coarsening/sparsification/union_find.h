#pragma once

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

template <typename T> class UnionFind {
public:
  UnionFind(T size);
  T find(T x);
  void unionNodes(T x, T y);

private:
  StaticArray<T> _parent;
  StaticArray<T> _rank;
};

} // namespace kaminpar::shm::sparsification
