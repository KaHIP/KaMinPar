#pragma once
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::coarsening::sparsification {
template <typename T> class UnionFind<T> {
public:
  UnionFind(T size);
  T find(T x);
  void unionNodes(T x, T y);

private:
  kaminpar::StaticArray<T> _parent;
  kaminpar::StaticArray<T> _rank;
};
} // namespace kaminpar::coarsening::sparsification
