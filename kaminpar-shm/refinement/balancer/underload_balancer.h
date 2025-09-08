/*******************************************************************************
 * MultiQueue-based balancer for greedy minimum block weight balancing.
 *
 * @file:   underload_balancer.h
 * @author: Daniel Seemaier
 * @date:   11.06.2025
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/multi_queue.h"

namespace kaminpar::shm {

class UnderloadBalancer : public Refiner {
  template <typename ConcretizedGraph>
  using GainCache = OnTheFlyGainCache<
      ConcretizedGraph,
      /*iterate_nonadjacent_blocks=*/true,
      /*iterate_exact_gains=*/true>;

public:
  explicit UnderloadBalancer(const Context &ctx);

  ~UnderloadBalancer() override;

  UnderloadBalancer &operator=(const UnderloadBalancer &) = delete;
  UnderloadBalancer(const UnderloadBalancer &) = delete;

  UnderloadBalancer &operator=(UnderloadBalancer &&) = delete;
  UnderloadBalancer(UnderloadBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  template <typename Graph> void rebalance_worker(const Graph &graph, int task_id);

  void init_underloaded_blocks();

  template <typename Graph> void init_pqs(const Graph &graph);

  [[nodiscard]] bool is_movable_to(BlockID block, NodeWeight node_weight) const;
  [[nodiscard]] bool is_movable_from(BlockID from, NodeWeight node_weight) const;

  std::pair<BlockID, float>
  compute_best_gain(const auto &graph, auto &gain_cache, NodeID node, BlockID from);

  void insert_node_into_pq(NodeID node, BlockID to, float gain);

  void lock_block(BlockID block);
  void unlock_block(BlockID block);

  const Context &_ctx;

  const PartitionContext *_p_ctx;
  PartitionedGraph *_p_graph;

  std::vector<std::uint8_t> _is_underloaded;
  std::vector<std::uint8_t> _block_locks;

  MaxMultiQueue<NodeID, float> _mq;

  StaticArray<BlockID> _node_target;
  AnyGraphComponent<GainCache> _gain_cache;
};

} // namespace kaminpar::shm
