/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#pragma once

#include <cmath>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/refiner.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/binary_heap.h"
#include "common/random.h"

namespace kaminpar::shm {
namespace fm {
struct Stats {
  parallel::Atomic<NodeID> num_touched_nodes = 0;
  parallel::Atomic<NodeID> num_committed_moves = 0;
  parallel::Atomic<NodeID> num_discarded_moves = 0;
  parallel::Atomic<NodeID> num_recomputed_gains = 0;
  parallel::Atomic<NodeID> num_batches = 0;
  parallel::Atomic<NodeID> num_pq_inserts = 0;
  parallel::Atomic<NodeID> num_pq_updates = 0;
  parallel::Atomic<NodeID> num_pq_pops = 0;

  Stats &operator+=(const Stats &other) {
    num_touched_nodes += other.num_touched_nodes;
    num_committed_moves += other.num_committed_moves;
    num_discarded_moves += other.num_discarded_moves;
    num_recomputed_gains += other.num_recomputed_gains;
    num_batches += other.num_batches;
    num_pq_inserts += other.num_pq_inserts;
    num_pq_updates += other.num_pq_updates;
    num_pq_pops += other.num_pq_pops;
    return *this;
  }
};

struct GlobalStats {
  std::vector<Stats> iteration_stats;

  GlobalStats() {
    next_iteration();
  }

  void add(const Stats &stats) {
    iteration_stats.back() += stats;
  }

  void next_iteration() {
    iteration_stats.emplace_back();
  }

  void reset() {
    iteration_stats.clear();
    next_iteration();
  }

  void summarize() {
    LOG_STATS << "FM Refinement:";
    for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
      const Stats &stats = iteration_stats[i];
      if (stats.num_batches == 0) {
        continue;
      }

      LOG_STATS << "  * Iteration " << (i + 1) << ":";
      LOG_STATS << "    + Number of batches: " << stats.num_batches;
      LOG_STATS << "    + Number of touched nodes: " << stats.num_touched_nodes << " in total, "
                << 1.0 * stats.num_touched_nodes / stats.num_batches << " per batch";
      LOG_STATS << "    + Number of moves: " << stats.num_committed_moves << " committed, "
                << stats.num_discarded_moves << " discarded (= "
                << 100.0 * stats.num_discarded_moves /
                       (stats.num_committed_moves + stats.num_discarded_moves)
                << "%)";
      LOG_STATS << "    + Number of recomputed gains: " << stats.num_recomputed_gains;
      LOG_STATS << "    + Number of PQ operations: " << stats.num_pq_inserts << " inserts, "
                << stats.num_pq_updates << " updates, " << stats.num_pq_pops << " pops";
    }
  }
};

class NodeTracker {
public:
  static constexpr int UNLOCKED = 0;
  static constexpr int MOVED_LOCALLY = -1;
  static constexpr int MOVED_GLOBALLY = -2;

  NodeTracker(const NodeID max_n) : _state(max_n) {
    tbb::parallel_for<std::size_t>(0, max_n, [&](const std::size_t i) { _state[i] = 0; });
  }

  bool lock(const NodeID u, const int id) {
    int free = 0;
    return __atomic_compare_exchange_n(
        &_state[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
    );
  }

  int owner(const NodeID u) const {
    return __atomic_load_n(&_state[u], __ATOMIC_RELAXED);
  }

  // @todo Build a better interface once the details are settled.
  void set(const NodeID node, const int value) {
    __atomic_store_n(&_state[node], value, __ATOMIC_RELAXED);
  }

private:
  NoinitVector<int> _state;
};

class BorderNodes {
public:
  BorderNodes(DenseGainCache &gain_cache, NodeTracker &node_tracker)
      : _gain_cache(gain_cache),
        _node_tracker(node_tracker) {}

  void init(const PartitionedGraph &p_graph) {
    _border_nodes.clear();
    p_graph.pfor_nodes([&](const NodeID u) {
      if (_gain_cache.is_border_node(u, p_graph.block(u))) {
        _border_nodes.push_back(u);
      }
      _node_tracker.set(u, 0);
    });
    _next_border_node = 0;
  }

  template <typename Container>
  void init_precomputed(const PartitionedGraph &p_graph, const Container &border_nodes) {
    _border_nodes.clear();
    for (const auto &u : border_nodes) {
      _border_nodes.push_back(u);
    }
    p_graph.pfor_nodes([&](const NodeID u) { _node_tracker.set(u, 0); });
    _next_border_node = 0;
  }

  template <typename Lambda> NodeID poll(const NodeID count, int id, Lambda &&lambda) {
    NodeID polled = 0;

    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to = std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[current];
        if (_node_tracker.owner(node) == NodeTracker::UNLOCKED && _node_tracker.lock(node, id)) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  NodeID get() const {
    return has_more() ? _border_nodes[_next_border_node] : kInvalidNodeID;
  }

  [[nodiscard]] bool has_more() const {
    return _next_border_node < _border_nodes.size();
  }

  [[nodiscard]] std::size_t size() const {
    return _border_nodes.size();
  }

  void shuffle() {
    Random::instance().shuffle(_border_nodes.begin(), _border_nodes.end());
  }

private:
  DenseGainCache &_gain_cache;
  NodeTracker &_node_tracker;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
};

struct SharedData {
  SharedData(const NodeID max_n, const BlockID max_k)
      : node_tracker(max_n),
        gain_cache(max_k, max_n),
        border_nodes(gain_cache, node_tracker),
        shared_pq_handles(max_n),
        target_blocks(max_n) {
    tbb::parallel_for<std::size_t>(0, shared_pq_handles.size(), [&](std::size_t i) {
      shared_pq_handles[i] = SharedBinaryMaxHeap<EdgeWeight>::kInvalidID;
    });
  }

  NodeTracker node_tracker;
  DenseGainCache gain_cache;
  BorderNodes border_nodes;
  NoinitVector<std::size_t> shared_pq_handles;
  NoinitVector<BlockID> target_blocks;
  GlobalStats stats;
};
} // namespace fm

class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);
  ~FMRefiner(); // Required for the std::unique_ptr<> member.

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  std::unique_ptr<fm::SharedData> _shared;
};

class LocalizedFMRefiner {
public:
  struct Move {
    NodeID node;
    BlockID from;
  };

  LocalizedFMRefiner(
      int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      fm::SharedData &shared
  );

  EdgeWeight run_batch();

  void enable_move_recording();
  std::vector<Move> take_applied_moves();

private:
  template <typename PartitionedGraphType, typename GainCacheType>
  void insert_into_node_pq(
      const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, NodeID u
  );

  void update_after_move(NodeID node, NodeID moved_node, BlockID moved_from, BlockID moved_to);

  template <typename PartitionedGraphType, typename GainCacheType>
  std::pair<BlockID, EdgeWeight>
  best_gain(const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, NodeID u);

  bool update_block_pq();

  // Unique worker ID
  int _id;

  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  // Graph to work on
  PartitionedGraph &_p_graph;

  // Data shared among all workers
  fm::SharedData &_shared;

  // Data local to this worker
  DeltaPartitionedGraph _d_graph;                         // O(|Delta|) space
  DeltaGainCache<DenseGainCache> _d_gain_cache;           // O(|Delta|) space
  BinaryMaxHeap<EdgeWeight> _block_pq;                    // O(k) space
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pqs; // O(k + |Touched|) space

  AdaptiveStoppingPolicy _stopping_policy;

  std::vector<NodeID> _touched_nodes;
  std::vector<NodeID> _seed_nodes;

  std::vector<Move> _applied_moves;
  bool _record_applied_moves = false;
};
} // namespace kaminpar::shm
