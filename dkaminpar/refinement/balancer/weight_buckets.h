/*******************************************************************************
 * Weight buckets for overloaded blocks.
 *
 * @file:   weight_buckets.h
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 ******************************************************************************/
#pragma once

#include <algorithm>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/metrics.h"

namespace kaminpar::dist {
class Buckets {
  constexpr static std::size_t kNumBuckets = 16;

public:
  static std::size_t compute_bucket(const double gain) {
    if (gain > 0) {
      return 0;
    } else if (gain == 0) {
      return std::min<std::size_t>(kNumBuckets - 1, 1);
    } else { // gain < 0
      return std::min<std::size_t>(kNumBuckets - 1, 2 + std::floor(std::log2(-gain)));
    }
  }

  Buckets(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx)
      : _p_graph(p_graph),
        _p_ctx(p_ctx),
        _bucket_sizes(p_graph.k() * kNumBuckets) {
    clear();
  }

  Buckets(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      StaticArray<GlobalNodeWeight> compactified_sizes
  )
      : Buckets(p_graph, p_ctx) {
    BlockID compact_block = 0;
    for (const BlockID block : p_graph.blocks()) {
      if (p_graph.block_weight(block) > p_ctx.graph->max_block_weight(block)) {
        std::copy(
            compactified_sizes.begin() + compact_block * kNumBuckets,
            compactified_sizes.begin() + (compact_block + 1) * kNumBuckets,
            _bucket_sizes.begin() + block * kNumBuckets
        );
        ++compact_block;
      }
    }
  }

  void clear() {
    std::fill(_bucket_sizes.begin(), _bucket_sizes.end(), 0);
  }

  void add(const BlockID block, const double gain) {
    size(block, compute_bucket(gain)) += _p_graph.block_weight(block);
  }

  GlobalNodeWeight &size(const BlockID block, const std::size_t bucket) {
    return _bucket_sizes[block * kNumBuckets + bucket];
  }

  GlobalNodeWeight size(const BlockID block, const std::size_t bucket) const {
    return _bucket_sizes[block * kNumBuckets + bucket];
  }

  StaticArray<GlobalNodeWeight> compactify() const {
    const BlockID num_overloaded_blocks = metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
    StaticArray<GlobalNodeWeight> compactified_sizes(num_overloaded_blocks * kNumBuckets);
    BlockID compact_block = 0;

    for (const BlockID block : _p_graph.blocks()) {
      if (_p_graph.block_weight(block) > _p_ctx.graph->max_block_weight(block)) {
        std::copy(
            _bucket_sizes.begin() + block * kNumBuckets,
            _bucket_sizes.begin() + (block + 1) * kNumBuckets,
            compactified_sizes.begin() + compact_block * kNumBuckets
        );
        ++compact_block;
      }
    }

    return compactified_sizes;
  }

private:
  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  StaticArray<GlobalNodeWeight> _bucket_sizes;
};
} // namespace kaminpar::dist
