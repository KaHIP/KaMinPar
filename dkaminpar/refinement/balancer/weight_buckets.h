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
  Buckets(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const bool positive_buckets,
      const double base
  )
      : _p_graph(p_graph),
        _p_ctx(p_ctx),
        _positive_buckets(positive_buckets),
        _base(base),
        _num_buckets(compute_num_buckets(positive_buckets, base)),
        _bucket_sizes(p_graph.k() * _num_buckets) {
    clear();
  }

  Buckets(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const bool positive_buckets,
      const double base,
      StaticArray<GlobalNodeWeight> compactified
  )
      : Buckets(p_graph, p_ctx, positive_buckets, base) {
    BlockID cb = 0;
    for (const BlockID b : p_graph.blocks()) {
      if (p_graph.block_weight(b) > p_ctx.graph->max_block_weight(b)) {
        std::copy(
            compactified.begin() + cb * _num_buckets,
            compactified.begin() + (cb + 1) * _num_buckets,
            _bucket_sizes.begin() + b * _num_buckets
        );
        ++cb;
      }
    }
  }

  std::size_t compute_bucket(const double gain) const {
    if (gain > 0) {
      return _positive_buckets ? neutral_bucket() - std::ceil(std::log2(gain) / std::log2(_base))
                               : 0;
    } else if (gain == 0) {
      return neutral_bucket();
    } else { // gain < 0
      return neutral_bucket() + std::ceil(std::log2(-gain) / std::log2(_base));
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
    return _bucket_sizes[block * _num_buckets + bucket];
  }

  GlobalNodeWeight size(const BlockID block, const std::size_t bucket) const {
    return _bucket_sizes[block * _num_buckets + bucket];
  }

  StaticArray<GlobalNodeWeight> compactify() const {
    const BlockID num_overloaded_blocks = metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
    StaticArray<GlobalNodeWeight> compactified(num_overloaded_blocks * num_buckets());

    BlockID cb = 0;
    for (const BlockID b : _p_graph.blocks()) {
      if (_p_graph.block_weight(b) > _p_ctx.graph->max_block_weight(b)) {
        std::copy(
            _bucket_sizes.begin() + b * _num_buckets,
            _bucket_sizes.begin() + (b + 1) * _num_buckets,
            compactified.begin() + cb * _num_buckets
        );
        ++cb;
      }
    }

    return compactified;
  }

  std::size_t num_buckets() const {
    return _num_buckets;
  }

private:
  static std::size_t compute_num_buckets(const bool positive_buckets, const double base) {
    const std::size_t num_neg_buckets =
        std::ceil(1.0 * std::numeric_limits<EdgeWeight>::digits / std::log2(base));
    return num_neg_buckets + (positive_buckets ? num_neg_buckets : 1);
  }

  std::size_t neutral_bucket() const {
    return _positive_buckets ? _num_buckets / 2 : 1;
  }

  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  bool _positive_buckets;
  double _base;
  std::size_t _num_buckets;

  StaticArray<GlobalNodeWeight> _bucket_sizes;
};
} // namespace kaminpar::dist
