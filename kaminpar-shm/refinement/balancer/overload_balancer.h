/*******************************************************************************
 * Greedy balancing algorithms that uses one thread per overloaded block.
 *
 * @file:   overload_balancer.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/dynamic_de_heap.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class OverloadBalancer : public Refiner {
  template <typename ConcretizedGraph>
  using GainCache = OnTheFlyGainCache<
      ConcretizedGraph,
      /*iterate_nonadjacent_blocks=*/true,
      /*iterate_exact_gains=*/true>;

public:
  using MoveTracker = std::function<void(NodeID, BlockID, BlockID)>;

  explicit OverloadBalancer(const Context &ctx);

  ~OverloadBalancer() override;

  OverloadBalancer &operator=(const OverloadBalancer &) = delete;
  OverloadBalancer(const OverloadBalancer &) = delete;

  OverloadBalancer &operator=(OverloadBalancer &&) = delete;
  OverloadBalancer(OverloadBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void track_moves(MoveTracker move_tracker);

private:
  template <typename Graph> BlockWeight perform_round(const Graph &graph);

  template <typename Graph> void init_pq(const Graph &graph);

  std::pair<BlockID, float>
  compute_best_gain(const auto &graph, auto &gain_cache, NodeID node, BlockID from);

  bool add_to_pq(BlockID block, NodeID node, NodeWeight weight, float rel_gain);

  void init_feasible_target_blocks();
  [[nodiscard]] BlockWeight block_overload(BlockID b) const;

  bool move_to_random_block(NodeID node);

  bool move_node_if_possible(NodeID node, BlockID from, BlockID to);

  const Context &_ctx;

  const PartitionContext *_p_ctx;
  PartitionedGraph *_p_graph;

  DynamicBinaryMinMaxForest<NodeID, float, ScalableVector> _pq;
  std::vector<BlockWeight> _pq_weight;
  tbb::enumerable_thread_specific<std::vector<BlockID>> _feasible_target_blocks;

  StaticArray<std::uint8_t> _moved_nodes;

  AnyGraphComponent<GainCache> _gain_cache;

  MoveTracker _move_tracker = nullptr;
};

} // namespace kaminpar::shm
