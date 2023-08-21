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
public:
  constexpr static int kNumBuckets = 16;

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
      StaticArray<GlobalNodeWeight> compactified
  )
      : Buckets(p_graph, p_ctx) {
    BlockID cb = 0;
    for (const BlockID b : p_graph.blocks()) {
      if (p_graph.block_weight(b) > p_ctx.graph->max_block_weight(b)) {
        std::copy(
            compactified.begin() + cb * kNumBuckets,
            compactified.begin() + (cb + 1) * kNumBuckets,
            _bucket_sizes.begin() + b * kNumBuckets
        );
        ++cb;
      }
    }
  }

  void clear() {
    std::fill(_bucket_sizes.begin(), _bucket_sizes.end(), 0);
  }

  void add(const BlockID block, const NodeWeight weight, const double gain) {
    size(block, compute_bucket(gain)) += weight;
  }

  void remove(const BlockID block, const NodeWeight weight, const double gain) {
    KASSERT(
        size(block, compute_bucket(gain)) >= weight,
        "removing " << weight << " from bucket " << compute_bucket(gain) << " (cor. to gain "
                    << gain << ") in block " << block << " would underflow the bucket size "
                    << size(block, compute_bucket(gain))
    );
    size(block, compute_bucket(gain)) -= weight;
  }

  GlobalNodeWeight &size(const BlockID block, const std::size_t bucket) {
    return _bucket_sizes[block * kNumBuckets + bucket];
  }

  GlobalNodeWeight size(const BlockID block, const std::size_t bucket) const {
    return _bucket_sizes[block * kNumBuckets + bucket];
  }

  StaticArray<GlobalNodeWeight> compactify() const {
    const BlockID num_overloaded_blocks = metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
    StaticArray<GlobalNodeWeight> compactified(num_overloaded_blocks * kNumBuckets);

    BlockID cb = 0;
    for (const BlockID b : _p_graph.blocks()) {
      if (_p_graph.block_weight(b) > _p_ctx.graph->max_block_weight(b)) {
        std::copy(
            _bucket_sizes.begin() + b * kNumBuckets,
            _bucket_sizes.begin() + (b + 1) * kNumBuckets,
            compactified.begin() + cb * kNumBuckets
        );
        ++cb;
      }
    }

    return compactified;
  }

private:
  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  StaticArray<GlobalNodeWeight> _bucket_sizes;
};
} // namespace kaminpar::dist
