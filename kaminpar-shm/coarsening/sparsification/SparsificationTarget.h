//
// Created by badger on 5/14/24.
//

#ifndef SPARSIFICATIONTARGET_H
#define SPARSIFICATIONTARGET_H
#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar {
namespace shm {
namespace sparsification {

class SparsificationTarget {
public:
  virtual EdgeID computeTarget(const Graph &oldGraph, NodeID newVertexAmount) = 0;
};

} // namespace sparsification
} // namespace shm
} // namespace kaminpar

#endif // SPARSIFICATIONTARGET_H
