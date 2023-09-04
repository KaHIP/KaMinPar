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

  Stats &operator+=(const Stats &other);
};

struct GlobalStats {
  std::vector<Stats> iteration_stats;

  GlobalStats();

  void add(const Stats &stats);
  void next_iteration();
  void reset();
  void summarize();
};

struct BatchStats {
  NodeID size;
  NodeID max_distance;
  std::vector<NodeID> size_by_distance;
  std::vector<EdgeWeight> gain_by_distance;
};

struct GlobalBatchStats {
  std::vector<std::vector<BatchStats>> iteration_stats;

  void next_iteration(std::vector<BatchStats> stats);
  void reset();
  void summarize();

private:
  void summarize_iteration(const std::size_t iteration, const std::vector<BatchStats> &stats);
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
  GlobalBatchStats batch_stats;
};

struct Move {
  NodeID node;
  BlockID from;
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
  using SeedNodesVec = std::vector<NodeID>;
  using MovesVec = std::vector<fm::Move>;
  using Batches = tbb::concurrent_vector<std::pair<SeedNodesVec, MovesVec>>;

  std::vector<fm::BatchStats>
  dbg_compute_batch_stats(const PartitionedGraph &graph, Batches next_batches) const;

  std::pair<PartitionedGraph, Batches>
  dbg_build_prev_p_graph(const PartitionedGraph &p_graph, Batches next_batches) const;

  fm::BatchStats dbg_compute_single_batch_stats_in_sequence(
      PartitionedGraph &p_graph,
      const std::vector<NodeID> &seeds,
      const std::vector<fm::Move> &moves
  ) const;

  std::vector<NodeID> dbg_compute_batch_distances(
      const Graph &graph, const std::vector<NodeID> &seeds, const std::vector<fm::Move> &moves
  ) const;

  const Context &_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  std::unique_ptr<fm::SharedData> _shared;
};

class LocalizedFMRefiner {
public:
  LocalizedFMRefiner(
      int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      fm::SharedData &shared
  );

  EdgeWeight run_batch();

  void enable_move_recording();
  std::vector<fm::Move> take_applied_moves();
  const std::vector<fm::Move> &last_batch_moves();
  const std::vector<NodeID> &last_batch_seed_nodes();

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

  std::vector<fm::Move> _applied_moves;
  bool _record_applied_moves = false;
};
} // namespace kaminpar::shm
