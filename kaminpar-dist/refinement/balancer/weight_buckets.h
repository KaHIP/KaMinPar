/*******************************************************************************
 * Weight buckets for overloaded blocks.
 *
 * @file:   weight_buckets.h
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 ******************************************************************************/
#pragma once

#include <algorithm>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/metrics.h"

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
        _bucket_sizes(p_graph.k() * _num_buckets),
        _cutoff_buckets(p_graph.k()) {
    clear();
  }

  [[nodiscard]] std::size_t compute_bucket(const double gain) const {
    std::size_t bucket;
    if (gain > 0) {
      bucket = _positive_buckets
                   ? neutral_bucket() - std::ceil(std::log2(1 + gain) / std::log2(_base))
                   : 0;
    } else if (gain == 0) {
      bucket = neutral_bucket();
    } else { // gain < 0
      bucket = neutral_bucket() + std::ceil(std::log2(1 - gain) / std::log2(_base));
    }

    KASSERT(bucket < num_buckets());
    return bucket;
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

  [[nodiscard]] GlobalNodeWeight size(const BlockID block, const std::size_t bucket) const {
    return _bucket_sizes[block * _num_buckets + bucket];
  }

  [[nodiscard]] StaticArray<GlobalNodeWeight> compactify() const {
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

  [[nodiscard]] std::size_t num_buckets() const {
    return _num_buckets;
  }

  const StaticArray<std::size_t> &
  compute_cutoff_buckets(const StaticArray<BlockWeight> &compactified_buckets) {
    const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

    std::vector<int> compactified_cutoff_buckets(metrics::num_imbalanced_blocks(_p_graph, _p_ctx));
    if (BlockID compactified_block = 0; mpi::get_comm_rank(_p_graph.communicator()) == 0) {
      for (const BlockID block : _p_graph.blocks()) {
        BlockWeight current_weight = _p_graph.block_weight(block);
        const BlockWeight max_weight = _p_ctx.graph->max_block_weight(block);

        if (current_weight > max_weight) {
          int cutoff_bucket = 0;
          for (; cutoff_bucket < num_buckets() && current_weight > max_weight; ++cutoff_bucket) {
            current_weight -=
                compactified_buckets[compactified_block * num_buckets() + cutoff_bucket];
          }

          KASSERT(compactified_block < compactified_cutoff_buckets.size());
          compactified_cutoff_buckets[compactified_block++] = cutoff_bucket;
        }
      }
    }

    // Broadcast to other PEs
    MPI_Bcast(
        compactified_cutoff_buckets.data(),
        asserting_cast<int>(compactified_cutoff_buckets.size()),
        MPI_INT,
        0,
        _p_graph.communicator()
    );

    for (BlockID compactified_block = 0, block = 0; block < _p_graph.k(); ++block) {
      BlockWeight current_weight = _p_graph.block_weight(block);
      const BlockWeight max_weight = _p_ctx.graph->max_block_weight(block);
      if (current_weight > max_weight) {
        _cutoff_buckets[block] = compactified_cutoff_buckets[compactified_block++];
      } else {
        _cutoff_buckets[block] = 0;
      }
    }

    return _cutoff_buckets;
  }

private:
  static std::size_t compute_num_buckets(const bool positive_buckets, const double base) {
    const std::size_t num_neg_buckets =
        std::ceil(1.0 * std::numeric_limits<double>::digits / std::log2(base));
    return num_neg_buckets + positive_buckets * num_neg_buckets + 1;
  }

  [[nodiscard]] std::size_t neutral_bucket() const {
    return _positive_buckets ? _num_buckets / 2 : 1;
  }

  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  bool _positive_buckets;
  double _base;
  std::size_t _num_buckets;

  StaticArray<GlobalNodeWeight> _bucket_sizes;
  StaticArray<std::size_t> _cutoff_buckets;
};
} // namespace kaminpar::dist
