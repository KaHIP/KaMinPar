/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/fm/fm_definitions.h"
#include "kaminpar-shm/refinement/fm/stopping_policies.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {
std::unique_ptr<Refiner> create_fm_refiner(const Context &ctx);

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
  void print();
};

class NodeTracker {
public:
  static constexpr int UNLOCKED = 0;
  static constexpr int MOVED_LOCALLY = -1;
  static constexpr int MOVED_GLOBALLY = -2;

  NodeTracker(const NodeID max_n) : _state(max_n) {}

  bool lock(const NodeID u, const int id) {
    int free = 0;
    return __atomic_compare_exchange_n(
        &_state[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
    );
  }

  [[nodiscard]] int owner(const NodeID u) const {
    return __atomic_load_n(&_state[u], __ATOMIC_RELAXED);
  }

  void set(const NodeID node, const int value) {
    __atomic_store_n(&_state[node], value, __ATOMIC_RELAXED);
  }

  void reset() {
    tbb::parallel_for<NodeID>(0, _state.size(), [&](const NodeID node) {
      _state[node] = UNLOCKED;
    });
  }

  void free() {
    _state.free();
  }

private:
  StaticArray<int> _state;
};

template <typename GainCache> class BorderNodes {
public:
  BorderNodes(const Context &ctx, GainCache &gain_cache, NodeTracker &node_tracker)
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

  [[nodiscard]] NodeID get() const {
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
  GainCache &_gain_cache;
  NodeTracker &_node_tracker;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
};

template <typename GainCache> struct SharedData {
  SharedData(const Context &ctx, const NodeID preallocate_n, const BlockID preallocate_k)
      : node_tracker(preallocate_n),
        gain_cache(ctx, preallocate_n, preallocate_k),
        border_nodes(ctx, gain_cache, node_tracker),
        shared_pq_handles(preallocate_n, SharedBinaryMaxHeap<EdgeWeight>::kInvalidID),
        target_blocks(static_array::noinit, preallocate_n) {}

  SharedData(const SharedData &) = delete;
  SharedData &operator=(const SharedData &) = delete;

  SharedData(SharedData &&) noexcept = default;
  SharedData &operator=(SharedData &&) = delete;

  ~SharedData() {
    tbb::parallel_invoke(
        [&] { shared_pq_handles.free(); },
        [&] { target_blocks.free(); },
        [&] { node_tracker.free(); },
        [&] { gain_cache.free(); }
    );
  }

  NodeTracker node_tracker;
  GainCache gain_cache;
  BorderNodes<GainCache> border_nodes;
  StaticArray<std::size_t> shared_pq_handles;
  StaticArray<BlockID> target_blocks;
  GlobalStats stats;
};
} // namespace fm

template <typename GainCache, typename DeltaPartitionedGraph = GenericDeltaPartitionedGraph<>>
class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);
  ~FMRefiner() override; // Required for the std::unique_ptr<> member.

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  std::unique_ptr<fm::SharedData<GainCache>> _shared;
};

template <typename GainCache, typename DeltaPartitionedGraph = GenericDeltaPartitionedGraph<>>
class LocalizedFMRefiner {
public:
  LocalizedFMRefiner(
      int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      fm::SharedData<GainCache> &shared
  );

  EdgeWeight run_batch();

  void enable_move_recording();
  const std::vector<fm::AppliedMove> &last_batch_moves();
  const std::vector<NodeID> &last_batch_seed_nodes();

private:
  template <typename GainCacheType, typename PartitionedGraphType>
  void insert_into_node_pq(
      const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, NodeID u
  );

  void update_after_move(NodeID node, NodeID moved_node, BlockID moved_from, BlockID moved_to);

  template <typename GainCacheType, typename PartitionedGraphType>
  std::pair<BlockID, EdgeWeight>
  best_gain(const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, NodeID u);

  bool update_block_pq();

  // Unique worker ID
  int _id;

  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  // Shared: Graph to work on
  PartitionedGraph &_p_graph;

  // Shared: Data shared among all workers
  fm::SharedData<GainCache> &_shared;

  // Thread-local: O(|Delta|) space
  DeltaPartitionedGraph _d_graph;

  // Thread local: O(|Delta|) sparse
  using DeltaGainCache = typename GainCache::template DeltaCache<DeltaPartitionedGraph>;
  DeltaGainCache _d_gain_cache;

  // Thread local: O(k) space
  BinaryMaxHeap<EdgeWeight> _block_pq;

  // Thread local: O(k + |Touched|) space
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pqs;

  AdaptiveStoppingPolicy _stopping_policy;

  // Thread local: O(|Touched|) space
  std::vector<NodeID> _touched_nodes;

  // Thread local: O(1) space
  std::vector<NodeID> _seed_nodes;

  // Thread local: O(|Touched|) space if move recording is enabled
  std::vector<fm::AppliedMove> _applied_moves;
  bool _record_applied_moves = false;
};
} // namespace kaminpar::shm
