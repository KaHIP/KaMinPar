#pragma once

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
class IndexDistributionWithoutReplacement {
public:
  IndexDistributionWithoutReplacement(std::vector<double> values) {
    IndexDistributionWithoutReplacement(values.begin(), values.end());
  }
  IndexDistributionWithoutReplacement(StaticArray<double> values) {
    IndexDistributionWithoutReplacement(values.begin(), values.end());
  }
  template <typename Iterator>
  IndexDistributionWithoutReplacement(Iterator values_begin, Iterator values_end)
      : remaining_objects(values_end - values_begin) {
    if (remaining_objects == 0)
      return;

    // size of a complete binary tree, where all values can be in the leaves
    size_t size = 1;
    while (size <= 2 * remaining_objects) {
      size *= 2;
    }
    size -= 1;
    segment_tree.resize(size, 0);

    // initalize leafs
    const size_t first_leaf = firstLeaf();
    for (size_t leaf = first_leaf; leaf < first_leaf + remaining_objects; leaf++) {
      segment_tree[leaf] = *(values_begin + (leaf - first_leaf));
    }

    // calculate sum of subtrees
    for (size_t node = segment_tree.size() - 1; node != 0; node--) {
      segment_tree[parent(node)] += segment_tree[node];
    }
  }

  size_t operator()() {
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

    size_t index = to_index(current_subtree);
    double value = segment_tree[current_subtree];

    // delete
    while (current_subtree != 0) {
      segment_tree[current_subtree] -= value;
      current_subtree = parent(current_subtree);
    }
    segment_tree[0] -= value;

    remaining_objects--;
    return index;
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
  size_t to_index(size_t leaf) {
    return leaf - firstLeaf();
  }
  std::vector<double> segment_tree;
  size_t remaining_objects;
};
} // namespace kaminpar::shm::sparsification
