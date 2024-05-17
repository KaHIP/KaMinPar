//
// Created by badger on 5/14/24.
//

#pragma once
#include "kaminpar-shm/coarsening/sparsification/SparsificationTarget.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::sparsification {
class EdgeReductionSparsificationTarget : public SparsificationTarget {
  float _factor;

public:
  EdgeReductionSparsificationTarget(float factor) : _factor(factor) {}
  EdgeID computeTarget(const Graph &oldGraph, NodeID newVertexAmount) override {
    return _factor * oldGraph.m();
  }
};
} // namespace kaminpar::shm::sparsification
