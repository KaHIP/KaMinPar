/*******************************************************************************
 * Distributed JET balancer due to: "Jet: Multilevel Graph Partitioning on GPUs"
 * by Gilbert et al.
 *
 * @file:   jet_balancer.cc
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 ******************************************************************************/
#include "dkaminpar/refinement/jet/jet_balancer.h"

#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/metrics.h"

#include "common/timer.h"

namespace kaminpar::dist {
JetBalancerFactory::JetBalancerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
JetBalancerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<JetBalancer>(_ctx, p_graph, p_ctx);
}

JetBalancer::JetBalancer(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _gain_calculator(_p_graph),
      _local_buckets(_p_graph, _p_ctx) {}

void JetBalancer::initialize() {}

bool JetBalancer::refine() {
  using namespace jet;

  _local_buckets.init(_gain_calculator);

  for (int it = 0; it < num_weak_iterations(); ++it) {
    weak_iteration();
  }
  for (int it = 0; it < num_strong_iterations(); ++it) {
    strong_iteration();
  }

  return false;
}

void JetBalancer::weak_iteration() {}

void JetBalancer::strong_iteration() {}

StaticArray<GlobalNodeWeight> JetBalancer::compute_compacitifed_global_bucket_sizes(
    StaticArray<GlobalNodeWeight> local_bucket_sizes
) {
  MPI_Allreduce(
      MPI_IN_PLACE,
      local_bucket_sizes.data(),
      local_bucket_sizes.size(),
      mpi::type::get<GlobalNodeWeight>(),
      MPI_SUM,
      MPI_COMM_WORLD
  );
  return local_bucket_sizes;
}

bool JetBalancer::is_overloaded_block(const BlockID block) const {
  return block_overload(block) > 0;
}

BlockWeight JetBalancer::block_overload(const BlockID block) const {
  return std::max<BlockWeight>(
      0, _p_ctx.graph->max_block_weight(block) - _p_graph.block_weight(block)
  );
}

int JetBalancer::num_weak_iterations() const {
  return _ctx.refinement.jet_balancer.num_weak_iterations;
}

int JetBalancer::num_strong_iterations() const {
  return _ctx.refinement.jet_balancer.num_strong_iterations;
}

namespace jet {
Buckets::Buckets(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx)
    : _p_graph(p_graph),
      _p_ctx(p_ctx),
      _bucket_sizes(p_graph.k() * kNumBuckets) {
  clear();
}

Buckets::Buckets(
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

void Buckets::init(const GainCalculator &gain_calculator) {
  for (const NodeID node : _p_graph.nodes()) {
    const BlockID block = _p_graph.block(node);
    if (_p_graph.block_weight(block) > _p_ctx.graph->max_block_weight(block)) {
      const EdgeWeight gain = gain_calculator.compute_absolute_gain(node, _p_ctx).first;
      const std::size_t bucket = compute_bucket(gain);
      size(block, bucket) += _p_graph.node_weight(node);
    }
  }
}

void Buckets::clear() {
  std::fill(_bucket_sizes.begin(), _bucket_sizes.end(), 0);
}

GlobalNodeWeight &Buckets::size(const BlockID block, const std::size_t bucket) {
  return _bucket_sizes[block * kNumBuckets + bucket];
}

GlobalNodeWeight Buckets::size(const BlockID block, const std::size_t bucket) const {
  return _bucket_sizes[block * kNumBuckets + bucket];
}

StaticArray<GlobalNodeWeight> Buckets::compactify() const {
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

std::size_t Buckets::compute_bucket(const EdgeWeight gain) {
  if (gain > 0) {
    return 0;
  } else if (gain == 0) {
    return std::min<std::size_t>(kNumBuckets - 1, 1);
  } else { // gain < 0
    return std::min<std::size_t>(kNumBuckets - 1, 2 + std::floor(std::log2(-gain)));
  }
}
} // namespace jet
} // namespace kaminpar::dist
