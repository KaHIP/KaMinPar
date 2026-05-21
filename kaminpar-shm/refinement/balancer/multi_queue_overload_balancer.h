/*******************************************************************************
 * MultiQueue-based greedy overload balancing.
 *
 * @file:   multi_queue_overload_balancer.h
 * @author: Daniel Seemaier
 * @date:   29.04.2026
 ******************************************************************************/
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <tbb/task_group.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/shared_binary_heap.h"
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

  template <typename Graph, typename GainCacheT>
  bool refine_with_gain_cache(
      PartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const Graph &graph,
      GainCacheT &gain_cache
  );

  void track_moves(MoveTracker move_tracker);

private:
  enum NodeState : std::uint8_t {
    INACTIVE = 0,
    MOVABLE = 1,
    LOCKED = 2,
    MOVED = 3,
  };

  using PQ = SharedBinaryMaxHeap<float>;

  struct AccessToken {
    explicit AccessToken(int seed, std::size_t num_pqs);

    [[nodiscard]] std::size_t pick_random_pq();

    [[nodiscard]] std::array<std::size_t, 2> pick_two_random_pqs();

    std::mt19937 rng;
    std::uniform_int_distribution<std::size_t> dist;
  };

  struct Move {
    NodeID node = kInvalidNodeID;
    BlockID from = kInvalidBlockID;
    BlockID to = kInvalidBlockID;
    float gain = std::numeric_limits<float>::lowest();
  };

  bool begin_refinement(PartitionedGraph &p_graph, const PartitionContext &p_ctx);

  void finish_refinement();

  template <typename Graph, typename GainCacheT>
  void run_refinement(const Graph &graph, GainCacheT &gain_cache);

  template <typename Graph, typename GainCacheT>
  void init_pqs(const Graph &graph, GainCacheT &gain_cache);

  template <typename Graph, typename GainCacheT>
  void rebalance_worker(const Graph &graph, GainCacheT &gain_cache, int task_id);

  template <typename Graph>
  bool find_next_move(const Graph &graph, auto &gain_cache, AccessToken &token, Move &move);

  template <typename Graph>
  bool
  locked_extract_candidate(std::size_t pq_id, const Graph &graph, auto &gain_cache, Move &move);

  template <typename Graph> void update_neighbors(const Graph &graph, auto &gain_cache, Move move);

  std::pair<BlockID, float>
  compute_best_gain(const auto &graph, auto &gain_cache, NodeID node, BlockID from);

  std::pair<BlockID, float> compute_best_gain_of_candidates(
      const auto &graph,
      const auto &gain_cache,
      NodeID node,
      BlockID from,
      std::array<BlockID, 3> candidates
  );

  void insert_node_into_pq(NodeID node, BlockID to, float gain, AccessToken &token);

  void update_node_in_pq(NodeID node, float gain);

  void remove_node_from_pq(NodeID node);

  void init_overloaded_blocks();

  [[nodiscard]] bool is_overloaded(BlockID block) const;

  void deactivate_overloaded_block(BlockID block);

  [[nodiscard]] bool try_lock_node(NodeID node);

  void unlock_node(NodeID node);

  void mark_node_moved(NodeID node);

  void mark_node_inactive(NodeID node);

  [[nodiscard]] BlockWeight block_overload(BlockID block) const;

  bool move_node_if_possible(NodeID node, BlockID from, BlockID to);

  [[nodiscard]] bool try_lock_pq(std::size_t pq);

  void lock_pq(std::size_t pq);

  void unlock_pq(std::size_t pq);

  [[nodiscard]] float pq_top_key(std::size_t pq) const;

  void update_pq_top_key(std::size_t pq);

  void clear_pqs();

  const Context &_ctx;

  const PartitionContext *_p_ctx = nullptr;
  PartitionedGraph *_p_graph = nullptr;

  std::vector<std::uint8_t> _is_overloaded;
  std::atomic<std::size_t> _num_overloaded_blocks = 0;

  std::vector<PQ> _pqs;
  std::vector<std::uint8_t> _pq_locks;
  std::vector<float> _pq_top_keys;

  StaticArray<BlockID> _node_target;
  StaticArray<std::size_t> _node_pq;
  StaticArray<std::size_t> _pq_handles;
  StaticArray<std::uint8_t> _node_state;

  AnyGraphComponent<GainCache> _gain_cache;

  MoveTracker _move_tracker = nullptr;
};

template <typename Graph, typename GainCacheT>
bool MultiQueueOverloadBalancer::refine_with_gain_cache(
    PartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const Graph &graph,
    GainCacheT &gain_cache
) {
  if (!begin_refinement(p_graph, p_ctx)) {
    return false;
  }

  run_refinement(graph, gain_cache);
  finish_refinement();

  return true;
}

template <typename Graph, typename GainCacheT>
void MultiQueueOverloadBalancer::run_refinement(const Graph &graph, GainCacheT &gain_cache) {
  init_pqs(graph, gain_cache);

  tbb::task_group tg;
  for (int task_id = 0; task_id < _ctx.parallel.num_threads; ++task_id) {
    tg.run([&, task_id] { rebalance_worker(graph, gain_cache, task_id); });
  }
  tg.wait();
}

} // namespace kaminpar::shm
