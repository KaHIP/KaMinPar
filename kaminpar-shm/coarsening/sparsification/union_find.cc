#include "kaminpar-shm/coarsening/sparsification/union_find.h"

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::sparsification {

template <typename T> UnionFind<T>::UnionFind(T size) : _parent(size), _rank(size) {
  for (T i = 0; i < size; i++) {
    _parent[i] = i;
    _rank[i] = 0;
  }
}

template <typename T> T UnionFind<T>::find(T x) {
  if (_parent[x] == x)
    return x;
  return _parent[x] = find(_parent[x]);
}

template <typename T> void UnionFind<T>::unionNodes(T x, T y) {
  x = find(x);
  y = find(y);
  if (x == y)
    return;

  if (_rank[x] < _rank[y]) {
    _parent[x] = y;
  } else if (_rank[y] < _rank[x]) {
    _parent[y] = x;
  } else {
    _parent[y] = x;
    _rank[x]++;
  }
}

template class UnionFind<NodeID>;

} // namespace kaminpar::shm::sparsification
