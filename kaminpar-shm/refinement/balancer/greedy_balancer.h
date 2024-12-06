/*******************************************************************************
 * Greedy balancing algorithms that uses one thread per overloaded block.
 *
 * @file:   greedy_balancer.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename Graph> class GreedyBalancerImpl;

struct GreedyBalancerMemoryContext {
  DynamicBinaryMinMaxForest<NodeID, double, StaticArray> pq;
  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> rating_map;
  tbb::enumerable_thread_specific<std::vector<BlockID>> feasible_target_blocks;
  StaticArray<std::uint8_t> moved_nodes;
  std::vector<BlockWeight> pq_weight;
  NormalSparseGainCache<Graph> *gain_cache = nullptr;
};

class GreedyBalancer : public Refiner {
  using GreedyBalancerCSRImpl = GreedyBalancerImpl<CSRGraph>;
  using GreedyBalancerCompressedImpl = GreedyBalancerImpl<CompressedGraph>;

public:
  GreedyBalancer(const Context &ctx);
  ~GreedyBalancer() override;

  GreedyBalancer &operator=(const GreedyBalancer &) = delete;
  GreedyBalancer(const GreedyBalancer &) = delete;

  GreedyBalancer &operator=(GreedyBalancer &&) = default;
  GreedyBalancer(GreedyBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void track_moves(NormalSparseGainCache<Graph> *gain_cache);

private:
  std::unique_ptr<GreedyBalancerCSRImpl> _csr_impl;
  std::unique_ptr<GreedyBalancerCompressedImpl> _compressed_impl;

  GreedyBalancerMemoryContext _memory_context;
};

} // namespace kaminpar::shm
