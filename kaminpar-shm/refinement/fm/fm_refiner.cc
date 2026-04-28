/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/fm/fm_refiner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/multi_queue_overload_balancer.h"
#include "kaminpar-shm/refinement/fm/batch_stats.h"
#include "kaminpar-shm/refinement/fm/border_nodes.h"
#include "kaminpar-shm/refinement/fm/general_stats.h"
#include "kaminpar-shm/refinement/fm/node_tracker.h"
#include "kaminpar-shm/refinement/fm/stopping_policies.h"

#include "kaminpar-common/datastructures/shared_binary_heap.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

// Gain cache variations: unless compiled with experimental features enabled, only the sparse gain
// cache will be available
#ifdef KAMINPAR_EXPERIMENTAL
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#endif

#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();

} // namespace

namespace fm {

enum class SearchMode : std::uint8_t {
  CONSTRAINED,
  UNCONSTRAINED,
};

struct GlobalMove {
  NodeID node;
  BlockID from;
  BlockID to;
  bool valid = true;
};

class UnconstrainedFMData {
  static constexpr std::uint32_t kNumBuckets = 16;
  static constexpr double kBucketFactor = 1.5;

public:
  UnconstrainedFMData(const NodeID preallocate_n, const BlockID preallocate_k)
      : _bucket_weights(preallocate_k * kNumBuckets),
        _virtual_weight_delta(preallocate_k),
        _rebalancing_nodes(preallocate_n) {}

  template <typename Graph, typename GainCache>
  void initialize(
      const KwayFMRefinementContext &fm_ctx,
      const Graph &graph,
      const PartitionedGraph &p_graph,
      const GainCache &gain_cache
  ) {
    if (_current_k != p_graph.k()) {
      _current_k = p_graph.k();
      _bucket_weights.resize(_current_k * kNumBuckets);
      _virtual_weight_delta.resize(_current_k);
    }
    if (_rebalancing_nodes.size() < graph.n()) {
      _rebalancing_nodes.resize(graph.n());
    }

    reset(graph.n());

    tbb::enumerable_thread_specific<std::vector<BlockWeight>> local_bucket_weights([&] {
      return std::vector<BlockWeight>(_current_k * kNumBuckets);
    });
    tbb::enumerable_thread_specific<std::unordered_map<std::uint64_t, BlockWeight>>
        local_fallback_bucket_weights;

    graph.pfor_nodes([&](const NodeID u) {
      const NodeWeight node_weight = graph.node_weight(u);
      if (node_weight == 0) {
        return;
      }

      EdgeWeight total_incident_weight = 0;
      graph.adjacent_nodes(u, [&](const NodeID, const EdgeWeight weight) {
        total_incident_weight += weight;
      });

      const BlockID block = p_graph.block(u);
      const EdgeWeight internal_weight = gain_cache.conn(u, block);
      if (static_cast<double>(internal_weight) <
          fm_ctx.unconstrained_rebalancing_node_inclusion_threshold * total_incident_weight) {
        return;
      }

      const std::uint32_t bucket =
          bucket_for_gain_per_weight(static_cast<double>(internal_weight) / node_weight);
      _rebalancing_nodes[u] = 1;
      if (bucket < kNumBuckets) {
        local_bucket_weights.local()[index(block, bucket)] += node_weight;
      } else {
        local_fallback_bucket_weights.local()[fallback_key(block, bucket - kNumBuckets)] +=
            node_weight;
      }
    });

    _bucket_weights.assign(_bucket_weights.size(), 0);
    for (const std::vector<BlockWeight> &local_weights : local_bucket_weights) {
      for (std::size_t i = 0; i < local_weights.size(); ++i) {
        _bucket_weights[i] += local_weights[i];
      }
    }

    tbb::parallel_for<BlockID>(0, _current_k, [&](const BlockID block) {
      for (std::uint32_t bucket = 0; bucket + 1 < kNumBuckets; ++bucket) {
        _bucket_weights[index(block, bucket + 1)] += _bucket_weights[index(block, bucket)];
      }
    });

    std::vector<std::unordered_map<std::uint32_t, BlockWeight>> fallback_weights_by_block(
        _current_k
    );
    for (const auto &local_weights : local_fallback_bucket_weights) {
      for (const auto &[key, weight] : local_weights) {
        const auto [block, bucket] = fallback_key_to_pair(key);
        fallback_weights_by_block[block][bucket] += weight;
      }
    }

    _fallback_bucket_weights.assign(_current_k, {});
    tbb::parallel_for<BlockID>(0, _current_k, [&](const BlockID block) {
      auto &weights_by_bucket = fallback_weights_by_block[block];
      if (weights_by_bucket.empty()) {
        return;
      }

      std::uint32_t max_bucket = 0;
      for (const auto &[bucket, weight] : weights_by_bucket) {
        max_bucket = std::max(max_bucket, bucket);
      }

      std::vector<BlockWeight> &fallback_weights = _fallback_bucket_weights[block];
      fallback_weights.resize(max_bucket + 1, 0);
      fallback_weights[0] += _bucket_weights[index(block, kNumBuckets - 1)];
      for (const auto &[bucket, weight] : weights_by_bucket) {
        fallback_weights[bucket] += weight;
      }
      for (std::uint32_t bucket = 0; bucket + 1 < fallback_weights.size(); ++bucket) {
        fallback_weights[bucket + 1] += fallback_weights[bucket];
      }
    });

    _initialized = true;
  }

  [[nodiscard]] bool initialized() const {
    return _initialized;
  }

  void free() {
    _bucket_weights.free();
    _virtual_weight_delta.free();
    _rebalancing_nodes.free();
    _fallback_bucket_weights.clear();
    _initialized = false;
    _current_k = kInvalidBlockID;
  }

  [[nodiscard]] bool is_rebalancing_node(const NodeID u) const {
    return _initialized && _rebalancing_nodes[u];
  }

  void add_virtual_weight_delta(const BlockID block, const BlockWeight delta) {
    __atomic_fetch_add(&_virtual_weight_delta[block], delta, __ATOMIC_RELAXED);
  }

  [[nodiscard]] BlockWeight virtual_weight_delta(const BlockID block) const {
    return __atomic_load_n(&_virtual_weight_delta[block], __ATOMIC_RELAXED);
  }

  [[nodiscard]] EdgeWeight estimate_penalty(
      const BlockID to, const BlockWeight initial_imbalance, const NodeWeight moved_weight
  ) const {
    KASSERT(_initialized);

    std::uint32_t bucket = 0;
    while (bucket < kNumBuckets &&
           initial_imbalance + moved_weight > _bucket_weights[index(to, bucket)]) {
      ++bucket;
    }

    if (bucket == kNumBuckets) {
      const std::vector<BlockWeight> &fallback_weights = _fallback_bucket_weights[to];
      while (bucket < kNumBuckets + fallback_weights.size() &&
             initial_imbalance + moved_weight > fallback_weights[bucket - kNumBuckets]) {
        ++bucket;
      }
    }

    return bucket == kNumBuckets + _fallback_bucket_weights[to].size()
               ? std::numeric_limits<EdgeWeight>::max()
               : static_cast<EdgeWeight>(
                     std::ceil(moved_weight * gain_per_weight_for_bucket(bucket))
                 );
  }

private:
  void reset(const NodeID n) {
    _bucket_weights.assign(_bucket_weights.size(), 0);
    _virtual_weight_delta.assign(_virtual_weight_delta.size(), 0);
    _fallback_bucket_weights.assign(_current_k, {});
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { _rebalancing_nodes[u] = 0; });
    _initialized = false;
  }

  [[nodiscard]] std::size_t index(const BlockID block, const std::uint32_t bucket) const {
    return static_cast<std::size_t>(block) * kNumBuckets + bucket;
  }

  [[nodiscard]] static std::uint64_t fallback_key(const BlockID block, const std::uint32_t bucket) {
    return (static_cast<std::uint64_t>(block) << 32) + bucket;
  }

  [[nodiscard]] static std::pair<BlockID, std::uint32_t>
  fallback_key_to_pair(const std::uint64_t key) {
    return {static_cast<BlockID>(key >> 32), static_cast<std::uint32_t>(key)};
  }

  [[nodiscard]] static double gain_per_weight_for_bucket(const std::uint32_t bucket) {
    if (bucket > 1) {
      return std::pow(kBucketFactor, bucket - 2);
    } else if (bucket == 1) {
      return 0.5;
    } else {
      return 0.0;
    }
  }

  [[nodiscard]] static std::uint32_t bucket_for_gain_per_weight(const double gain_per_weight) {
    if (gain_per_weight >= 1.0) {
      return static_cast<std::uint32_t>(
          2 + std::ceil(std::log(gain_per_weight) / std::log(kBucketFactor))
      );
    } else if (gain_per_weight > 0.5) {
      return 2;
    } else if (gain_per_weight > 0.0) {
      return 1;
    } else {
      return 0;
    }
  }

  bool _initialized = false;
  BlockID _current_k = kInvalidBlockID;
  StaticArray<BlockWeight> _bucket_weights;
  StaticArray<BlockWeight> _virtual_weight_delta;
  StaticArray<std::uint8_t> _rebalancing_nodes;
  std::vector<std::vector<BlockWeight>> _fallback_bucket_weights;
};

template <typename GainCache> struct SharedData {
  SharedData(const Context &ctx, const NodeID preallocate_n, const BlockID preallocate_k)
      : node_tracker(preallocate_n),
        gain_cache(ctx, preallocate_n, preallocate_k),
        border_nodes(gain_cache, node_tracker),
        shared_pq_handles(preallocate_n, SharedBinaryMaxHeap<EdgeWeight>::kInvalidID),
        target_blocks(preallocate_n, static_array::noinit),
        unconstrained(preallocate_n, preallocate_k) {}

  SharedData(const SharedData &) = delete;
  SharedData &operator=(const SharedData &) = delete;

  SharedData(SharedData &&) noexcept = default;
  SharedData &operator=(SharedData &&) = delete;

  ~SharedData() {
    tbb::parallel_invoke(
        [&] { shared_pq_handles.free(); },
        [&] { target_blocks.free(); },
        [&] { unconstrained.free(); },
        [&] { node_tracker.free(); },
        [&] { gain_cache.free(); }
    );
  }

  NodeTracker node_tracker;
  GainCache gain_cache;
  BorderNodes<GainCache> border_nodes;
  StaticArray<std::size_t> shared_pq_handles;
  StaticArray<BlockID> target_blocks;
  UnconstrainedFMData unconstrained;
  tbb::concurrent_vector<GlobalMove> round_moves;
  tbb::concurrent_vector<GlobalMove> rebalancing_moves;
  GlobalStatistics stats;

  std::atomic<std::uint8_t> abort = 0;
};

template <typename Graph, typename GainCache, SearchMode kSearchMode> class LocalizedFMRefiner {
  using DeltaGainCache = typename GainCache::DeltaGainCache;

  static constexpr bool kAllowOverloadedMoves = kSearchMode == SearchMode::UNCONSTRAINED;

public:
  LocalizedFMRefiner(
      const int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      const Graph &graph,
      PartitionedGraph &p_graph,
      fm::SharedData<GainCache> &shared
  )
      : _id(id),
        _p_ctx(p_ctx),
        _fm_ctx(fm_ctx),
        _graph(graph),
        _p_graph(p_graph),
        _shared(shared),
        _d_graph(&_p_graph),
        _d_gain_cache(_shared.gain_cache, _d_graph),
        _block_pq(_p_graph.k()),
        _stopping_policy(_fm_ctx.alpha) {
    _stopping_policy.init(_graph.n());
    for ([[maybe_unused]] const BlockID b : _p_graph.blocks()) {
      _node_pqs.emplace_back(_graph.n(), _shared.shared_pq_handles.data());
    }
  }

  void configure_round(
      const bool allow_overloaded_moves,
      const double unconstrained_penalty_factor,
      const double unconstrained_upper_bound
  ) {
    _allow_overloaded_moves = kAllowOverloadedMoves && allow_overloaded_moves;
    _unconstrained_penalty_factor = unconstrained_penalty_factor;
    _unconstrained_upper_bound = unconstrained_upper_bound;

    _local_virtual_weight_delta.resize(_p_graph.k());
    std::fill(_local_virtual_weight_delta.begin(), _local_virtual_weight_delta.end(), 0);
  }

  EdgeWeight run_batch() {
    using fm::NodeTracker;

    _seed_nodes.clear();
    _applied_moves.clear();
    _local_moves.clear();
    std::fill(_local_virtual_weight_delta.begin(), _local_virtual_weight_delta.end(), 0);

    // Statistics for this batch only, to be merged into the global stats
    fm::IterationStatistics stats;
    IFSTATS(stats(Statistic::NUM_BATCHES));

    // Poll seed nodes from the border node arrays
    _shared.border_nodes.poll(_fm_ctx.num_seed_nodes, _id, [&](const NodeID seed_node) {
      insert_into_node_pq(_p_graph, _shared.gain_cache, seed_node);
      _seed_nodes.push_back(seed_node);

      IFSTATS(stats(Statistic::NUM_TOUCHED_NODES));
    });

    // Keep track of the current (expected) gain to decide when to accept a delta partition
    EdgeWeight current_total_gain = 0;
    EdgeWeight best_total_gain = 0;

    int steps = 0;

    while (update_block_pq() && !_stopping_policy.should_stop() &&
           ((++steps %= 64) || !_shared.abort.load(std::memory_order_relaxed))) {
      const BlockID block_from = _block_pq.peek_id();
      KASSERT(block_from < _p_graph.k());

      const NodeID node = _node_pqs[block_from].peek_id();
      KASSERT(node < _graph.n());

      const EdgeWeight expected_gain = _node_pqs[block_from].peek_key();
      const auto [block_to, actual_gain] = find_best_gain(_d_graph, _d_gain_cache, node);

      // If the gain got worse, reject the move and try again
      if (actual_gain < expected_gain) {
        _node_pqs[block_from].change_priority(node, actual_gain);
        _shared.target_blocks[node] = block_to;
        if (_node_pqs[block_from].peek_key() != _block_pq.key(block_from)) {
          _block_pq.change_priority(block_from, _node_pqs[block_from].peek_key());
        }

        IFSTATS(stats(Statistic::NUM_RECOMPUTED_GAINS));
        continue;
      }

      // Otherwise, we can remove the node from the PQ
      _node_pqs[block_from].pop();
      _shared.node_tracker.set(node, NodeTracker::MOVED_LOCALLY);
      IFSTATS(stats(Statistic::NUM_PQ_POPS));

      // Skip the move if there is no viable target block
      if (block_to == block_from) {
        continue;
      }

      // Accept the move if the target block does not get overloaded, unless this is an
      // unconstrained search.
      const NodeWeight node_weight = _graph.node_weight(node);
      if (allow_overloaded_moves() ||
          _d_graph.block_weight(block_to) + node_weight <= _p_ctx.max_block_weight(block_to)) {
        const BlockWeight from_weight_before = _d_graph.block_weight(block_from);
        const BlockWeight to_weight_before = _d_graph.block_weight(block_to);
        current_total_gain += actual_gain;

        const bool found_gain_improvement = current_total_gain > best_total_gain;
        const bool found_balance_improvement =
            current_total_gain == best_total_gain &&
            from_weight_before == heaviest_block_weight(_d_graph) &&
            to_weight_before + node_weight < from_weight_before;

        // If we found a new local minimum, apply the moves to the global
        // partition
        if (found_gain_improvement || found_balance_improvement) {
          for (const GlobalMove &move : _local_moves) {
            apply_global_move(move);

            if (_record_applied_moves) {
              _applied_moves.push_back(
                  fm::AppliedMove{
                      .node = move.node,
                      .from = move.from,
                      .improvement = false,
                  }
              );
            }

            IFSTATS(stats(Statistic::NUM_COMMITTED_MOVES));
          }

          apply_global_move({.node = node, .from = block_from, .to = block_to, .valid = true});
          record_unconstrained_move(node, block_from);
          flush_unconstrained_move_data();

          if (_record_applied_moves) {
            _applied_moves.push_back(
                fm::AppliedMove{
                    .node = node,
                    .from = block_from,
                    .improvement = true,
                }
            );
          }
          IFSTATS(stats(Statistic::NUM_COMMITTED_MOVES));

          // Flush local delta
          _d_graph.clear();
          _d_gain_cache.clear();
          _local_moves.clear();
          _stopping_policy.reset();

          best_total_gain = current_total_gain;
        } else {
          // Perform local move
          _d_graph.set_block(node, block_to);
          _d_gain_cache.move(node, block_from, block_to);
          _local_moves.push_back({.node = node, .from = block_from, .to = block_to, .valid = true});
          record_unconstrained_move(node, block_from);
          _stopping_policy.update(actual_gain);
        }

        _graph.adjacent_nodes(node, [&](const NodeID v) {
          const int owner = _shared.node_tracker.owner(v);
          if (owner == _id) {
            KASSERT(_node_pqs[_p_graph.block(v)].contains(v), "owned node not in PQ");
            IFSTATS(stats(Statistic::NUM_PQ_UPDATES));

            update_after_move(v, node, block_from, block_to);
          } else if (owner == NodeTracker::UNLOCKED && _shared.node_tracker.lock(v, _id)) {
            IFSTATS(stats(Statistic::NUM_PQ_INSERTS));
            IFSTATS(stats(Statistic::NUM_TOUCHED_NODES));

            insert_into_node_pq(_d_graph, _d_gain_cache, v);
            _touched_nodes.push_back(v);
          }
        });
      }
    }

    // Flush local state for the nex tround
    for (auto &node_pq : _node_pqs) {
      node_pq.clear();
    }

    auto unlock_touched_node = [&](const NodeID node) {
      const int owner = _shared.node_tracker.owner(node);
      if (owner == NodeTracker::MOVED_LOCALLY) {
        if (_fm_ctx.unlock_locally_moved_nodes) {
          _shared.node_tracker.set(node, NodeTracker::UNLOCKED);
        } else {
          _shared.node_tracker.set(node, NodeTracker::MOVED_GLOBALLY);
        }
      } else if (owner == _id) {
        _shared.node_tracker.set(node, NodeTracker::UNLOCKED);
      }
    };

    // If we do not wish to unlock seed nodes, mark them as globally moved == locked for good
    for (const NodeID &seed_node : _seed_nodes) {
      if (!_fm_ctx.unlock_seed_nodes) {
        _shared.node_tracker.set(seed_node, NodeTracker::MOVED_GLOBALLY);
      } else {
        unlock_touched_node(seed_node);
      }
    }

    // Unlock all nodes that were touched but not moved, or nodes that were only moved in the
    // thread-local delta graph
    IFSTATS(stats(Statistic::NUM_DISCARDED_MOVES, _d_graph.size()));

    for (const NodeID touched_node : _touched_nodes) {
      unlock_touched_node(touched_node);
    }

    _block_pq.clear();
    _d_graph.clear();
    _d_gain_cache.clear();
    _local_moves.clear();
    _stopping_policy.reset();
    _touched_nodes.clear();

    IFSTATS(_shared.stats.add(stats));
    return best_total_gain;
  }

  void enable_move_recording() {
    _record_applied_moves = true;
  }

  [[nodiscard]] const std::vector<fm::AppliedMove> &last_batch_moves() const {
    return _applied_moves;
  }

  [[nodiscard]] const std::vector<NodeID> &last_batch_seed_nodes() const {
    return _seed_nodes;
  }

private:
  // Note: p_graph could be a PartitionedGraph or a DeltaPartitionedGraph. If it is a
  // PartitionedGraph, gain_cache will be of type GainCache; if it is a DeltaPartitionedGraph,
  // gain_cache will also be a DeltaGainCache.
  void insert_into_node_pq(const auto &p_graph, const auto &gain_cache, const NodeID u) {
    const BlockID block_u = p_graph.block(u);
    const auto [block_to, gain] = find_best_gain(p_graph, gain_cache, u);

    KASSERT(block_u < _node_pqs.size(), "block_u out of bounds");
    KASSERT(!_node_pqs[block_u].contains(u), "node already contained in PQ");

    _shared.target_blocks[u] = block_to;
    _node_pqs[block_u].push(u, gain);
  }

  void update_after_move(
      const NodeID node,
      [[maybe_unused]] const NodeID moved_node,
      const BlockID moved_from,
      const BlockID moved_to
  ) {
    const BlockID old_block = _p_graph.block(node);
    const BlockID old_target_block = _shared.target_blocks[node];

    if (allow_overloaded_moves()) {
      const auto [new_target_block, new_gain] = find_best_gain(_d_graph, _d_gain_cache, node);
      _shared.target_blocks[node] = new_target_block;
      _node_pqs[old_block].change_priority(node, new_gain);
      return;
    }

    if (moved_to == old_target_block) {
      // In this case, old_target_block got even better
      // We only need to consider other blocks if old_target_block is full now
      if (_d_graph.block_weight(old_target_block) + _d_graph.node_weight(node) <=
          _p_ctx.max_block_weight(old_target_block)) {
        _node_pqs[old_block].change_priority(
            node, _d_gain_cache.gain(node, old_block, old_target_block)
        );
      } else {
        const auto [new_target_block, new_gain] = find_best_gain(_d_graph, _d_gain_cache, node);
        _shared.target_blocks[node] = new_target_block;
        _node_pqs[old_block].change_priority(node, new_gain);
      }
    } else if (moved_from == old_target_block) {
      // old_target_block go worse, thus have to re-consider all other blocks
      const auto [new_target_block, new_gain] = find_best_gain(_d_graph, _d_gain_cache, node);
      _shared.target_blocks[node] = new_target_block;
      _node_pqs[old_block].change_priority(node, new_gain);
    } else if (moved_to == old_block) {
      KASSERT(moved_from != old_target_block);

      // Since we did not move from old_target_block, this block is still the
      // best and we can still move to that block
      _node_pqs[old_block].change_priority(
          node, _d_gain_cache.gain(node, old_block, old_target_block)
      );
    } else {
      // old_target_block OR moved_to is best
      const auto [gain_old_target_block, gain_moved_to] =
          _d_gain_cache.gain(node, old_block, {old_target_block, moved_to});

      const bool moved_to_is_feasible =
          _d_graph.block_weight(moved_to) + _d_graph.node_weight(node) <=
          _p_ctx.max_block_weight(moved_to);
      if (gain_moved_to > gain_old_target_block && moved_to_is_feasible) {
        _shared.target_blocks[node] = moved_to;
        _node_pqs[old_block].change_priority(node, gain_moved_to);
      } else {
        _node_pqs[old_block].change_priority(node, gain_old_target_block);
      }
    }
  }

  // Note: p_graph could be a PartitionedGraph or a DeltaPartitionedGraph. If it is a
  // PartitionedGraph, gain_cache will be of type GainCache; if it is a DeltaPartitionedGraph,
  // gain_cache will also be a DeltaGainCache.
  std::pair<BlockID, EdgeWeight>
  find_best_gain(const auto &p_graph, const auto &gain_cache, const NodeID u) {
    const BlockID from = p_graph.block(u);
    const NodeWeight weight = _graph.node_weight(u);

    // Since we use max heaps, it is OK to insert this value into the PQ
    EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
    EdgeWeight best_penalty = 0;
    BlockID best_target_block = from;
    BlockWeight best_target_block_weight_gap =
        _p_ctx.max_block_weight(from) - p_graph.block_weight(from);

    gain_cache.gains(u, from, [&](const BlockID to, auto &&compute_gain) {
      const BlockWeight target_block_weight = p_graph.block_weight(to) + weight;
      const BlockWeight max_block_weight = _p_ctx.max_block_weight(to);
      const BlockWeight block_weight_gap = max_block_weight - target_block_weight;
      if (!allow_overloaded_moves()) {
        if (block_weight_gap < std::min<BlockWeight>(best_target_block_weight_gap, 0)) {
          return;
        }
      }

      EdgeWeight penalty = 0;
      if (allow_overloaded_moves()) {
        if (_unconstrained_upper_bound >= 1.0 &&
            target_block_weight > _unconstrained_upper_bound * max_block_weight) {
          return;
        }

        if (target_block_weight > max_block_weight && _unconstrained_penalty_factor > 0.0) {
          const EdgeWeight estimated_penalty = estimate_unconstrained_penalty(p_graph, to, weight);
          if (estimated_penalty == std::numeric_limits<EdgeWeight>::max()) {
            return;
          }

          const double scaled_penalty =
              std::ceil(_unconstrained_penalty_factor * estimated_penalty);
          if (scaled_penalty >= static_cast<double>(std::numeric_limits<EdgeWeight>::max())) {
            return;
          }

          penalty = static_cast<EdgeWeight>(scaled_penalty);
        }
      }

      const EdgeWeight gain = compute_gain() - penalty;
      if (gain > best_gain ||
          (gain == best_gain && block_weight_gap > best_target_block_weight_gap)) {
        best_gain = gain;
        best_penalty = penalty;
        best_target_block = to;
        best_target_block_weight_gap = block_weight_gap;
      }
    });

    const EdgeWeight actual_best_gain = [&] {
      if (best_target_block == from) {
        return std::numeric_limits<EdgeWeight>::min();
      } else {
        if constexpr (GainCache::kIteratesExactGains) {
          return best_gain;
        } else {
          return gain_cache.gain(u, from, best_target_block) - best_penalty;
        }
      }
    }();

    return {best_target_block, actual_best_gain};
  }

  [[nodiscard]] bool allow_overloaded_moves() const {
    return kAllowOverloadedMoves && _allow_overloaded_moves;
  }

  void apply_global_move(const GlobalMove &move) {
    KASSERT(_p_graph.block(move.node) == move.from);

    _shared.gain_cache.move(move.node, move.from, move.to);
    _shared.node_tracker.set(move.node, NodeTracker::MOVED_GLOBALLY);
    _p_graph.set_block(move.node, move.to);

    if constexpr (kAllowOverloadedMoves) {
      _shared.round_moves.push_back(move);
    }
  }

  [[nodiscard]] BlockWeight heaviest_block_weight(const auto &p_graph) const {
    BlockWeight heaviest = std::numeric_limits<BlockWeight>::min();
    for (const BlockID block : _p_graph.blocks()) {
      heaviest = std::max(heaviest, p_graph.block_weight(block));
    }
    return heaviest;
  }

  void record_unconstrained_move(const NodeID node, const BlockID from) {
    if (allow_overloaded_moves() && _shared.unconstrained.is_rebalancing_node(node)) {
      _local_virtual_weight_delta[from] += _graph.node_weight(node);
    }
  }

  void flush_unconstrained_move_data() {
    if (!allow_overloaded_moves()) {
      return;
    }

    for (BlockID block = 0; block < _local_virtual_weight_delta.size(); ++block) {
      const BlockWeight delta = _local_virtual_weight_delta[block];
      if (delta > 0) {
        _shared.unconstrained.add_virtual_weight_delta(block, delta);
        _local_virtual_weight_delta[block] = 0;
      }
    }
  }

  [[nodiscard]] EdgeWeight estimate_unconstrained_penalty(
      const auto &p_graph, const BlockID to, const NodeWeight weight
  ) const {
    const BlockWeight virtual_delta =
        _shared.unconstrained.virtual_weight_delta(to) + _local_virtual_weight_delta[to];
    const BlockWeight initial_imbalance =
        p_graph.block_weight(to) + virtual_delta - _p_ctx.max_block_weight(to);

    return _shared.unconstrained.estimate_penalty(to, initial_imbalance, weight);
  }

  bool update_block_pq() {
    bool have_more_nodes = false;
    for (const BlockID block : _p_graph.blocks()) {
      if (!_node_pqs[block].empty()) {
        const EdgeWeight gain = _node_pqs[block].peek_key();
        _block_pq.push_or_change_priority(block, gain);
        have_more_nodes = true;
      } else if (_block_pq.contains(block)) {
        _block_pq.remove(block);
      }
    }
    return have_more_nodes;
  }

  // Unique worker ID
  int _id;

  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  // Shared data structures:
  const Graph &_graph;
  PartitionedGraph &_p_graph;
  fm::SharedData<GainCache> &_shared;

  // Thread-local data structures:
  DeltaPartitionedGraph _d_graph;
  DeltaGainCache _d_gain_cache;

  BinaryMaxHeap<EdgeWeight> _block_pq;
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pqs;
  AdaptiveStoppingPolicy _stopping_policy;
  std::vector<NodeID> _touched_nodes;
  std::vector<NodeID> _seed_nodes;
  std::vector<fm::AppliedMove> _applied_moves;
  std::vector<GlobalMove> _local_moves;
  std::vector<BlockWeight> _local_virtual_weight_delta;
  bool _allow_overloaded_moves = kAllowOverloadedMoves;
  double _unconstrained_penalty_factor = 0.0;
  double _unconstrained_upper_bound = 0.0;
  bool _record_applied_moves = false;
};

} // namespace fm

template <
    typename Graph,
    template <typename> typename GainCacheTemplate,
    fm::SearchMode kSearchMode>
class FMRefinerCore : public Refiner {
  using GainCache = GainCacheTemplate<Graph>;
  using GlobalMove = fm::GlobalMove;

  static constexpr bool kAllowOverloadedMoves = kSearchMode == fm::SearchMode::UNCONSTRAINED;

public:
  FMRefinerCore(const Context &ctx) : _ctx(ctx), _fm_ctx(ctx.refinement.kway_fm) {}

  void initialize([[maybe_unused]] const PartitionedGraph &p_graph) final {
    if (_uninitialized) {
      SCOPED_HEAP_PROFILER("FM Allocation");
      _shared =
          std::make_unique<fm::SharedData<GainCache>>(_ctx, p_graph.graph().n(), _ctx.partition.k);
      _uninitialized = false;
    }
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final {
    SCOPED_HEAP_PROFILER("FM");
    SCOPED_TIMER("FM");

    const Graph &graph = concretize<Graph>(p_graph.graph());

    TIMED_SCOPE("Initialize gain cache") {
      _shared->gain_cache.initialize(graph, p_graph);
    };

    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
    EdgeWeight best_cut = initial_cut;
    EdgeWeight cut_before_current_iteration = initial_cut;
    EdgeWeight total_expected_gain = 0;
    bool last_iteration_is_best = true;

    StaticArray<BlockID> best_partition;
    StaticArray<BlockID> round_start_partition;
    if constexpr (kAllowOverloadedMoves) {
      best_partition.resize(graph.n());
      round_start_partition.resize(graph.n());
      graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
    }

    MultiQueueOverloadBalancer balancer(_ctx);
    if constexpr (kAllowOverloadedMoves) {
      balancer.initialize(p_graph);
      balancer.track_moves([&](const NodeID u, const BlockID from, const BlockID to) {
        _shared->gain_cache.move(u, from, to);
        _shared->rebalancing_moves.push_back({.node = u, .from = from, .to = to, .valid = true});
      });
    }

    // Create thread-local workers numbered 1..P
    using Worker = fm::LocalizedFMRefiner<Graph, GainCache, kSearchMode>;

    std::atomic<int> next_id = 0;
    tbb::enumerable_thread_specific<std::unique_ptr<Worker>> localized_fm_refiner_ets([&] {
      // It is important that worker IDs start at 1, otherwise the node
      // tracker won't work
      std::unique_ptr<Worker> localized_refiner =
          std::make_unique<Worker>(++next_id, p_ctx, _fm_ctx, graph, p_graph, *_shared);

      // If we want to evaluate the successful batches, record moves that are applied to the
      // global graph
      IF_STATSC(_fm_ctx.dbg_compute_batch_stats) {
        localized_refiner->enable_move_recording();
      }

      return localized_refiner;
    });

    fm::BatchStatsComputator batch_stats(p_graph);

    bool unconstrained_enabled = kAllowOverloadedMoves;

    auto interpolate_unconstrained_penalty = [&](const int iteration) {
      if (_fm_ctx.unconstrained_num_iterations <= 1) {
        return _fm_ctx.unconstrained_penalty_min;
      }

      const double start = _fm_ctx.unconstrained_penalty_min;
      const double end = _fm_ctx.unconstrained_penalty_max;
      return ((_fm_ctx.unconstrained_num_iterations - iteration - 1) * start + iteration * end) /
             static_cast<double>(_fm_ctx.unconstrained_num_iterations - 1);
    };

    for (int iteration = 0; iteration < _fm_ctx.num_iterations; ++iteration) {
      const bool use_unconstrained_iteration = kAllowOverloadedMoves && unconstrained_enabled &&
                                               iteration < _fm_ctx.unconstrained_num_iterations;
      const double unconstrained_penalty_factor =
          use_unconstrained_iteration ? interpolate_unconstrained_penalty(iteration) : 0.0;

      if constexpr (kAllowOverloadedMoves) {
        _shared->round_moves.clear();
        _shared->rebalancing_moves.clear();
        graph.pfor_nodes([&](const NodeID u) { round_start_partition[u] = p_graph.block(u); });

        if (use_unconstrained_iteration) {
          TIMED_SCOPE("Initialize unconstrained FM data") {
            _shared->unconstrained.initialize(_fm_ctx, graph, p_graph, _shared->gain_cache);
          };
        }
      }

      // Gains of the current iterations
      tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

      // Find current border nodes
      START_TIMER("Initialize border nodes");
      _shared->border_nodes.init(p_graph); // also resets the NodeTracker
      _shared->border_nodes.shuffle();
      _shared->abort = 0;
      STOP_TIMER();

      DBG << "Starting FM iteration " << iteration << " with " << _shared->border_nodes.size()
          << " border nodes and " << _ctx.parallel.num_threads << " worker threads";

      // Start one worker per thread
      if (graph.n() == _ctx.partition.n) {
        START_TIMER("Localized searches, fine level");
      } else {
        START_TIMER("Localized searches, coarse level");
      }

      std::atomic<int> num_finished_workers = 0;

      tbb::parallel_for<int>(0, _ctx.parallel.num_threads, [&](int) {
        auto &expected_gain = expected_gain_ets.local();
        auto &localized_refiner = *localized_fm_refiner_ets.local();
        localized_refiner.configure_round(
            use_unconstrained_iteration,
            unconstrained_penalty_factor,
            _fm_ctx.unconstrained_upper_bound
        );

        // The workers attempt to extract seed nodes from the border nodes
        // that are still available, continuing this process until there are
        // no more border nodes
        while (_shared->border_nodes.has_more()) {
          if (_fm_ctx.dbg_report_progress) {
            LLOG << " " << _shared->border_nodes.remaining();
          }

          const auto expected_batch_gain = localized_refiner.run_batch();
          expected_gain += expected_batch_gain;

          // Copies of the seed nodes and the moves are intentional: postpone actual stats
          // computation until after the FM iteration has finished
          IFSTATSC(
              _fm_ctx.dbg_compute_batch_stats && expected_batch_gain > 0,
              batch_stats.track(
                  localized_refiner.last_batch_seed_nodes(), localized_refiner.last_batch_moves()
              )
          );
        }

        if (++num_finished_workers >= _fm_ctx.minimal_parallelism) {
          _shared->abort = 1;
        }
      });
      STOP_TIMER();

      IFSTATSC(_fm_ctx.dbg_compute_batch_stats, batch_stats.next_iteration());

      const EdgeWeight expected_gain_of_this_iteration = expected_gain_ets.combine(std::plus{});
      total_expected_gain += expected_gain_of_this_iteration;

      NodeWeight current_overload = 0;
      if constexpr (kAllowOverloadedMoves) {
        current_overload = metrics::total_overload(p_graph, p_ctx);
        if (use_unconstrained_iteration && current_overload > 0) {
          TIMED_SCOPE("Rebalance") {
            balancer.refine(p_graph, p_ctx);
          };
          current_overload = metrics::total_overload(p_graph, p_ctx);
        }

        if (use_unconstrained_iteration) {
          interleave_rebalancing_moves(graph, p_graph, p_ctx, round_start_partition);
        }
      }

      EdgeWeight current_cut = kAllowOverloadedMoves || _fm_ctx.use_exact_abortion_threshold
                                   ? metrics::edge_cut(p_graph)
                                   : cut_before_current_iteration - expected_gain_of_this_iteration;

      if constexpr (kAllowOverloadedMoves) {
        TIMED_SCOPE("Rollback") {
          current_cut = rollback_to_best_prefix(
              graph, p_graph, p_ctx, round_start_partition, cut_before_current_iteration
          );
          _shared->gain_cache.initialize(graph, p_graph);
        };

        graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
        best_cut = current_cut;
        last_iteration_is_best = true;
      }

      const EdgeWeight abs_improvement_of_this_iteration =
          cut_before_current_iteration - current_cut;
      const double improvement_of_this_iteration =
          cut_before_current_iteration > 0
              ? 1.0 * abs_improvement_of_this_iteration / cut_before_current_iteration
              : 0.0;

      const bool switch_to_constrained_fm =
          use_unconstrained_iteration && _fm_ctx.unconstrained_min_improvement >= 0.0 &&
          improvement_of_this_iteration < _fm_ctx.unconstrained_min_improvement;
      if (switch_to_constrained_fm) {
        unconstrained_enabled = false;
      } else if (1.0 - improvement_of_this_iteration > _fm_ctx.abortion_threshold) {
        break;
      }

      cut_before_current_iteration = current_cut;
      DBG << "Expected gain of iteration " << iteration << ": " << expected_gain_of_this_iteration
          << ", total expected gain so far: " << total_expected_gain;
      IFSTATS(_shared->stats.next_iteration());
    }

    IFSTATS(_shared->stats.print());
    IFSTATS(_shared->stats.reset());
    IFSTATS(_shared->gain_cache.print_statistics());
    IFSTATSC(_fm_ctx.dbg_compute_batch_stats, batch_stats.print());

    if constexpr (kAllowOverloadedMoves) {
      TIMED_SCOPE("Rollback") {
        if (!last_iteration_is_best) {
          graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, best_partition[u]); });
          _shared->gain_cache.initialize(graph, p_graph);
        }
      };

      return best_cut < initial_cut;
    } else {
      return false;
    }
  }

private:
  [[nodiscard]] EdgeWeight compute_move_gain(
      const Graph &graph,
      const PartitionedGraph &p_graph,
      const NodeID node,
      const BlockID from,
      const BlockID to
  ) const {
    EdgeWeight conn_from = 0;
    EdgeWeight conn_to = 0;

    graph.adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      const BlockID block = p_graph.block(v);
      if (block == from) {
        conn_from += weight;
      } else if (block == to) {
        conn_to += weight;
      }
    });

    return conn_to - conn_from;
  }

  void interleave_rebalancing_moves(
      const Graph &graph,
      const PartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const StaticArray<BlockID> &round_start_partition
  ) const {
    if (_shared->rebalancing_moves.empty()) {
      return;
    }

    std::vector<GlobalMove> fm_moves(_shared->round_moves.begin(), _shared->round_moves.end());
    std::vector<GlobalMove> rebalancing_moves(
        _shared->rebalancing_moves.begin(), _shared->rebalancing_moves.end()
    );

    constexpr std::size_t kInvalidMoveIndex = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> move_index_of_node(graph.n(), kInvalidMoveIndex);
    for (std::size_t i = 0; i < fm_moves.size(); ++i) {
      const GlobalMove &move = fm_moves[i];
      if (move.valid) {
        move_index_of_node[move.node] = i;
      }
    }

    for (GlobalMove &rebalancing_move : rebalancing_moves) {
      if (!rebalancing_move.valid) {
        continue;
      }

      const std::size_t fm_move_index = move_index_of_node[rebalancing_move.node];
      if (fm_move_index == kInvalidMoveIndex) {
        continue;
      }

      GlobalMove &fm_move = fm_moves[fm_move_index];
      if (!fm_move.valid || fm_move.to != rebalancing_move.from) {
        continue;
      }

      if (fm_move.from == rebalancing_move.to) {
        fm_move.valid = false;
        rebalancing_move.valid = false;
      } else {
        rebalancing_move.from = fm_move.from;
        fm_move.valid = false;
      }
    }

    std::vector<std::vector<GlobalMove>> rebalancing_moves_by_block(p_graph.k());
    for (const GlobalMove &move : rebalancing_moves) {
      if (move.valid) {
        rebalancing_moves_by_block[move.from].push_back(move);
      }
    }

    std::vector<BlockWeight> block_weights(p_graph.k(), 0);
    for (const NodeID node : graph.nodes()) {
      block_weights[round_start_partition[node]] += graph.node_weight(node);
    }

    std::vector<std::size_t> next_rebalancing_move(p_graph.k(), 0);
    std::vector<GlobalMove> interleaved_moves;
    interleaved_moves.reserve(fm_moves.size() + rebalancing_moves.size());

    auto apply_to_block_weights = [&](const GlobalMove &move) {
      const NodeWeight weight = graph.node_weight(move.node);
      block_weights[move.from] -= weight;
      block_weights[move.to] += weight;
    };

    auto insert_moves_to_balance_block = [&](auto &self, const BlockID block) -> void {
      auto &moves = rebalancing_moves_by_block[block];
      std::size_t &next = next_rebalancing_move[block];

      while (block_weights[block] > p_ctx.max_block_weight(block) && next < moves.size()) {
        const GlobalMove move = moves[next++];
        interleaved_moves.push_back(move);
        apply_to_block_weights(move);

        if (block_weights[move.to] > p_ctx.max_block_weight(move.to)) {
          self(self, move.to);
        }
      }
    };

    for (const BlockID block : p_graph.blocks()) {
      insert_moves_to_balance_block(insert_moves_to_balance_block, block);
    }

    for (const GlobalMove &move : fm_moves) {
      if (!move.valid) {
        continue;
      }

      interleaved_moves.push_back(move);
      apply_to_block_weights(move);
      insert_moves_to_balance_block(insert_moves_to_balance_block, move.to);
    }

    for (const BlockID block : p_graph.blocks()) {
      auto &moves = rebalancing_moves_by_block[block];
      std::size_t &next = next_rebalancing_move[block];
      while (next < moves.size()) {
        interleaved_moves.push_back(moves[next++]);
      }
    }

    _shared->round_moves.clear();
    for (const GlobalMove &move : interleaved_moves) {
      _shared->round_moves.push_back(move);
    }
  }

  [[nodiscard]] EdgeWeight rollback_to_best_prefix(
      const Graph &graph,
      PartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const StaticArray<BlockID> &round_start_partition,
      const EdgeWeight round_start_cut
  ) const {
    const std::size_t num_moves = _shared->round_moves.size();

    graph.pfor_nodes([&](const NodeID u) {
      const BlockID block = round_start_partition[u];
      if (p_graph.block(u) != block) {
        p_graph.set_block(u, block);
      }
    });

    std::vector<BlockWeight> block_weights(p_graph.k());
    std::size_t overloaded_blocks = 0;
    for (const BlockID block : p_graph.blocks()) {
      block_weights[block] = p_graph.block_weight(block);
      overloaded_blocks += block_weights[block] > p_ctx.max_block_weight(block);
    }

    EdgeWeight current_cut = round_start_cut;
    EdgeWeight best_cut = round_start_cut;
    BlockWeight best_heaviest_block_weight =
        *std::max_element(block_weights.begin(), block_weights.end());
    std::size_t best_prefix = 0;
    std::vector<std::uint8_t> applied_moves(num_moves, 0);

    auto update_block_weight = [&](const BlockID block, const BlockWeight new_weight) {
      const bool was_overloaded = block_weights[block] > p_ctx.max_block_weight(block);
      block_weights[block] = new_weight;
      const bool is_overloaded = block_weights[block] > p_ctx.max_block_weight(block);

      overloaded_blocks += is_overloaded && !was_overloaded;
      overloaded_blocks -= was_overloaded && !is_overloaded;
    };

    for (std::size_t i = 0; i < num_moves; ++i) {
      const GlobalMove move = _shared->round_moves[i];
      if (!move.valid) {
        continue;
      }
      if (p_graph.block(move.node) != move.from) {
        continue;
      }

      current_cut -= compute_move_gain(graph, p_graph, move.node, move.from, move.to);

      const NodeWeight weight = graph.node_weight(move.node);
      update_block_weight(move.from, block_weights[move.from] - weight);
      update_block_weight(move.to, block_weights[move.to] + weight);
      p_graph.set_block(move.node, move.to);
      applied_moves[i] = 1;

      if (overloaded_blocks == 0) {
        const BlockWeight heaviest_block_weight =
            *std::max_element(block_weights.begin(), block_weights.end());
        if (current_cut < best_cut ||
            (current_cut == best_cut && heaviest_block_weight < best_heaviest_block_weight)) {
          best_cut = current_cut;
          best_heaviest_block_weight = heaviest_block_weight;
          best_prefix = i + 1;
        }
      }
    }

    for (std::size_t i = num_moves; i-- > best_prefix;) {
      if (!applied_moves[i]) {
        continue;
      }

      const GlobalMove move = _shared->round_moves[i];
      if (p_graph.block(move.node) == move.to) {
        p_graph.set_block(move.node, move.from);
      }
    }

    _shared->round_moves.clear();
    return best_cut;
  }

  const Context &_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  bool _uninitialized = true;
  std::unique_ptr<fm::SharedData<GainCache>> _shared;
};

template <fm::SearchMode kSearchMode, typename Graph>
std::unique_ptr<Refiner> create_fm_core(const Context &ctx) {
  switch (ctx.refinement.kway_fm.gain_cache_strategy) {
  case GainCacheStrategy::SPARSE:
    return std::make_unique<FMRefinerCore<Graph, NormalSparseGainCache, kSearchMode>>(ctx);

#ifdef KAMINPAR_EXPERIMENTAL
  case GainCacheStrategy::COMPACT_HASHING_LARGE_K:
    return std::make_unique<FMRefinerCore<Graph, LargeKCompactHashingGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::SPARSE_LARGE_K:
    return std::make_unique<FMRefinerCore<Graph, LargeKSparseGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::HASHING:
    return std::make_unique<FMRefinerCore<Graph, NormalHashingGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::HASHING_LARGE_K:
    return std::make_unique<FMRefinerCore<Graph, LargeKHashingGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::DENSE:
    return std::make_unique<FMRefinerCore<Graph, NormalDenseGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::DENSE_LARGE_K:
    return std::make_unique<FMRefinerCore<Graph, LargeKDenseGainCache, kSearchMode>>(ctx);

  case GainCacheStrategy::ON_THE_FLY:
    return std::make_unique<FMRefinerCore<Graph, NormalOnTheFlyGainCache, kSearchMode>>(ctx);
#endif // KAMINPAR_EXPERIMENTAL

  default:
    LOG_WARNING << "The selected gain cache strategy '"
                << stringify_enum(ctx.refinement.kway_fm.gain_cache_strategy)
                << "' is not available in this build. Rebuild with experimental features enabled.";
    LOG_WARNING << "Using the default gain cache strategy '"
                << stringify_enum(GainCacheStrategy::COMPACT_HASHING) << "' instead.";
    [[fallthrough]];

  case GainCacheStrategy::COMPACT_HASHING:
    return std::make_unique<FMRefinerCore<Graph, NormalCompactHashingGainCache, kSearchMode>>(ctx);
  }
}

FMRefiner::FMRefiner(const Context &input_ctx) : _ctx(input_ctx) {}
FMRefiner::~FMRefiner() = default;

std::string FMRefiner::name() const {
  return "FM";
}

void FMRefiner::initialize(const PartitionedGraph &p_graph) {
  reified(p_graph, [&]<typename Graph>(Graph &) {
    _core = create_fm_core<fm::SearchMode::CONSTRAINED, Graph>(_ctx);
  });

  _core->initialize(p_graph);
}

bool FMRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  if (p_ctx.has_min_block_weights()) {
    LOG_WARNING << "FM refinement does not support min block weights. They will be ignored.";
  }

  return _core->refine(p_graph, p_ctx);
}

UnconstrainedFMRefiner::UnconstrainedFMRefiner(const Context &input_ctx) : _ctx(input_ctx) {}
UnconstrainedFMRefiner::~UnconstrainedFMRefiner() = default;

std::string UnconstrainedFMRefiner::name() const {
  return "Unconstrained FM";
}

void UnconstrainedFMRefiner::initialize(const PartitionedGraph &p_graph) {
  reified(p_graph, [&]<typename Graph>(Graph &) {
    _core = create_fm_core<fm::SearchMode::UNCONSTRAINED, Graph>(_ctx);
  });

  _core->initialize(p_graph);
}

bool UnconstrainedFMRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  if (p_ctx.has_min_block_weights()) {
    LOG_WARNING
        << "Unconstrained FM refinement does not support min block weights. They will be ignored.";
  }

  return _core->refine(p_graph, p_ctx);
}

} // namespace kaminpar::shm
