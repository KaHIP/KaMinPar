#pragma once

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
template <typename Object> class DistributionWithoutReplacement {
public:
  DistributionWithoutReplacement(std::vector<Object> objects, std::vector<double> values)
      : objects(objects),
        remaining_objects(values.size()) {
    if (values.size() == 0)
      return;

    // size of a complete binary tree, where all values can be in the leaves
    size_t size = 1;
    while (size <= 2 * values.size()) {
      size *= 2;
    }
    size -= 1;
    segment_tree.resize(size, 0);

    // initalize leafs
    const size_t first_leaf = firstLeaf();
    for (size_t leaf = first_leaf; leaf < first_leaf + values.size(); leaf++) {
      segment_tree[leaf] = values[leaf - first_leaf];
    }

    // calculate sum of subtrees
    for (size_t node = segment_tree.size() - 1; node != 0; node--) {
      segment_tree[parent(node)] += segment_tree[node];
    }
  }

  Object operator()() {
    double r = Random::instance().random_double() * segment_tree[0];

    size_t current_subtree = 0;
    while (not isLeaf(current_subtree)) {
      if (r <= segment_tree[leftChild(current_subtree)]) {
        current_subtree = leftChild(current_subtree);
      } else {
        r -= segment_tree[leftChild(current_subtree)];
        current_subtree = rightChild(current_subtree);
      }
    }

    Object obj = to_object(current_subtree);
    double value = segment_tree[current_subtree];

    // delete
    while (current_subtree != 0) {
      segment_tree[current_subtree] -= value;
      current_subtree = parent(current_subtree);
    }
    segment_tree[0] -= value;

    remaining_objects--;
    return obj;
  }

  size_t size() {
    return remaining_objects;
  }
  bool empty() {
    return remaining_objects == 0;
  }

private:
  bool isLeaf(size_t i) {
    return i >= firstLeaf();
  }
  size_t parent(size_t i) {
    return (i - 1) / 2;
  }
  size_t leftChild(size_t i) {
    return 2 * i + 1;
  }
  size_t rightChild(size_t i) {
    return 2 * i + 2;
  }
  size_t firstLeaf() {
    return segment_tree.size() / 2;
  }
  Object to_object(size_t leaf) {
    size_t index = leaf - firstLeaf();
    return objects[index];
  }
  std::vector<double> segment_tree;
  std::vector<Object> objects;
  size_t remaining_objects;
};
} // namespace kaminpar::shm::sparsification
