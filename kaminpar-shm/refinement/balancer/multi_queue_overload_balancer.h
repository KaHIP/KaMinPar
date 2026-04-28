/*******************************************************************************
 * MultiQueue-based greedy overload balancing.
 *
 * @file:   multi_queue_overload_balancer.h
 * @author: Daniel Seemaier
 * @date:   29.04.2026
 ******************************************************************************/
#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/multi_queue.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class MultiQueueOverloadBalancer : public Refiner {
  template <typename ConcretizedGraph>
  using GainCache = OnTheFlyGainCache<
      ConcretizedGraph,
      /*iterate_nonadjacent_blocks=*/true,
      /*iterate_exact_gains=*/true,
      /*iterate_source_block=*/false>;

public:
  using MoveTracker = std::function<void(NodeID, BlockID, BlockID)>;

  explicit MultiQueueOverloadBalancer(const Context &ctx);

  ~MultiQueueOverloadBalancer() override;

  MultiQueueOverloadBalancer &operator=(const MultiQueueOverloadBalancer &) = delete;
  MultiQueueOverloadBalancer(const MultiQueueOverloadBalancer &) = delete;

  MultiQueueOverloadBalancer &operator=(MultiQueueOverloadBalancer &&) = delete;
  MultiQueueOverloadBalancer(MultiQueueOverloadBalancer &&) noexcept = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void track_moves(MoveTracker move_tracker);

private:
  enum NodeState : std::uint8_t {
    INACTIVE = 0,
    MOVABLE = 1,
    LOCKED = 2,
    MOVED = 3,
  };

  template <typename Graph> void init_pqs(const Graph &graph);

  template <typename Graph> void rebalance_worker(const Graph &graph, int task_id);

  std::pair<BlockID, float>
  compute_best_gain(const auto &graph, auto &gain_cache, NodeID node, BlockID from);

  void insert_node_into_pq(NodeID node, BlockID to, float gain);

  void init_overloaded_blocks();

  [[nodiscard]] bool is_overloaded(BlockID block) const;

  void deactivate_overloaded_block(BlockID block);

  [[nodiscard]] bool try_lock_node(NodeID node);

  void unlock_node(NodeID node);

  void mark_node_moved(NodeID node);

  void mark_node_inactive(NodeID node);

  [[nodiscard]] BlockWeight block_overload(BlockID block) const;

  bool move_node_if_possible(NodeID node, BlockID from, BlockID to);

  const Context &_ctx;

  const PartitionContext *_p_ctx = nullptr;
  PartitionedGraph *_p_graph = nullptr;

  std::vector<std::uint8_t> _is_overloaded;
  std::atomic<std::size_t> _num_overloaded_blocks = 0;

  MaxMultiQueue<NodeID, float> _mq;

  StaticArray<BlockID> _node_target;
  StaticArray<std::uint8_t> _node_state;

  AnyGraphComponent<GainCache> _gain_cache;

  MoveTracker _move_tracker = nullptr;
};

} // namespace kaminpar::shm
