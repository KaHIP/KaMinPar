/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   label_propagation.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::lp {

enum class RatingMapStrategy {
  SINGLE_PHASE,
  TWO_PHASE,
  GROWING_HASH_TABLES,
};

enum class ActiveSetStrategy {
  NONE,
  GLOBAL,
  LOCAL,
};

enum class TieBreakingStrategy {
  GEOMETRIC,
  UNIFORM,
};

template <typename NodeID, typename ClusterID> struct Initialization {
  NodeID num_nodes;
  NodeID num_active_nodes;
  ClusterID num_clusters;
};

template <typename NodeID, typename ClusterID> struct Options {
  NodeID max_degree = std::numeric_limits<NodeID>::max();
  NodeID max_num_neighbors = std::numeric_limits<NodeID>::max();
  ClusterID desired_num_clusters = 0;
  RatingMapStrategy rating_map_strategy = RatingMapStrategy::SINGLE_PHASE;
  ActiveSetStrategy active_set_strategy = ActiveSetStrategy::NONE;
  TieBreakingStrategy tie_breaking_strategy = TieBreakingStrategy::GEOMETRIC;
  bool track_cluster_count = false;
  bool use_two_hop_clustering = false;
  bool use_actual_gain = false;
  bool relabel_before_second_phase = false;
  std::size_t rating_map_threshold = 10000;
};

template <typename NodeID, typename EdgeWeight> struct MoveCandidate {
  NodeID node;
  EdgeWeight gain;
};

template <typename NodeID, typename ClusterID, typename EdgeWeight> struct PassResult {
  NodeID processed_nodes = 0;
  NodeID moved_nodes = 0;
  ClusterID removed_clusters = 0;
  EdgeWeight expected_total_gain = 0;
};

template <typename NodeID, typename ClusterID, typename ClusterWeight, typename EdgeWeight>
struct ClusterSelectionState {
  Random &local_rand;
  NodeID u;
  typename std::type_identity<ClusterWeight>::type u_weight;
  ClusterID initial_cluster;
  ClusterWeight initial_cluster_weight;
  ClusterID best_cluster;
  EdgeWeight best_gain;
  ClusterWeight best_cluster_weight;
  EdgeWeight overall_best_gain;
  ClusterID current_cluster;
  EdgeWeight current_gain;
  ClusterWeight current_cluster_weight;
};

template <typename ClusterID, typename EdgeWeight> struct LocalClusterSelectionState {
  EdgeWeight best_gain;
  ClusterID best_cluster;
  EdgeWeight favored_cluster_gain;
  ClusterID favored_cluster;
};

template <typename NodeID> struct StatelessNeighborPolicy {
  [[nodiscard]] bool accept(const NodeID, const NodeID) const {
    return true;
  }

  [[nodiscard]] bool activate(const NodeID) const {
    return true;
  }

  [[nodiscard]] bool skip(const NodeID) const {
    return false;
  }
};

template <typename NodeID, typename ClusterID> class ExternalLabelArray {
public:
  using ClusterIDType = ClusterID;

  void init(StaticArray<ClusterID> &labels) {
    _labels = &labels;
  }

  void init_cluster(const NodeID node, const ClusterID cluster) {
    move_node(node, cluster);
  }

  [[nodiscard]] ClusterID cluster(const NodeID node) const {
    KASSERT(_labels != nullptr);
    KASSERT(node < _labels->size());
    return __atomic_load_n(&_labels->at(node), __ATOMIC_RELAXED);
  }

  void move_node(const NodeID node, const ClusterID cluster) {
    KASSERT(_labels != nullptr);
    KASSERT(node < _labels->size());
    __atomic_store_n(&_labels->at(node), cluster, __ATOMIC_RELAXED);
  }

  [[nodiscard]] ClusterID initial_cluster(const NodeID node) const {
    return node;
  }

private:
  StaticArray<ClusterID> *_labels = nullptr;
};

template <typename ClusterID, typename ClusterWeight> class RelaxedClusterWeightVector {
public:
  using ClusterWeightType = ClusterWeight;
  using ClusterWeights = StaticArray<ClusterWeight>;

  void allocate(const ClusterID num_clusters) {
    if (_cluster_weights.size() < num_clusters) {
      _cluster_weights.resize(num_clusters);
    }
  }

  void free() {
    _cluster_weights.free();
  }

  void setup(ClusterWeights cluster_weights) {
    _cluster_weights = std::move(cluster_weights);
  }

  ClusterWeights release() {
    return std::move(_cluster_weights);
  }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights[cluster] = weight;
  }

  [[nodiscard]] ClusterWeight cluster_weight(const ClusterID cluster) const {
    return __atomic_load_n(&_cluster_weights[cluster], __ATOMIC_RELAXED);
  }

  [[nodiscard]] ClusterWeight initial_cluster_weight(const ClusterID cluster) const {
    return cluster_weight(cluster);
  }

  bool move_cluster_weight(
      const ClusterID old_cluster,
      const ClusterID new_cluster,
      const ClusterWeight delta,
      const ClusterWeight max_weight
  ) {
    if (_cluster_weights[new_cluster] + delta <= max_weight) {
      __atomic_fetch_add(&_cluster_weights[new_cluster], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_cluster_weights[old_cluster], delta, __ATOMIC_RELAXED);
      return true;
    }

    return false;
  }

  void reassign_cluster_weights(
      const StaticArray<ClusterID> &mapping, const ClusterID num_new_clusters
  ) {
    ClusterWeights new_cluster_weights(num_new_clusters);

    tbb::parallel_for(
        tbb::blocked_range<ClusterID>(0, _cluster_weights.size()), [&](const auto &r) {
          for (ClusterID c = r.begin(); c != r.end(); ++c) {
            const ClusterWeight weight = _cluster_weights[c];

            if (weight != 0) {
              const ClusterID new_cluster = mapping[c] - 1;
              new_cluster_weights[new_cluster] = weight;
            }
          }
        }
    );

    _cluster_weights = std::move(new_cluster_weights);
  }

private:
  ClusterWeights _cluster_weights;
};

template <
    typename NodeID,
    typename ClusterID,
    typename EdgeWeight,
    typename RatingMap,
    typename GrowingRatingMap = DynamicRememberingFlatMap<ClusterID, EdgeWeight>,
    typename ConcurrentRatingMap = ConcurrentFastResetArray<EdgeWeight, ClusterID>,
    bool kEnableTwoPhase = true>
struct Workspace {
  static constexpr bool kSupportsTwoPhase = kEnableTwoPhase;

  using RatingMapType = RatingMap;
  using GrowingRatingMapType = GrowingRatingMap;
  using ConcurrentRatingMapType = ConcurrentRatingMap;

  tbb::enumerable_thread_specific<RatingMap> rating_map_ets;
  tbb::enumerable_thread_specific<GrowingRatingMap> growing_rating_map_ets;
  ConcurrentRatingMap concurrent_rating_map;
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_clusters_ets;
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_favored_clusters_ets;
  StaticArray<std::uint8_t> active;
  StaticArray<std::uint8_t> moved;
  StaticArray<ClusterID> favored_clusters;
  tbb::concurrent_vector<NodeID> second_phase_nodes;
  std::vector<LocalClusterSelectionState<ClusterID, EdgeWeight>> local_cluster_selection_states;

  void allocate(
      const NodeID num_nodes,
      const NodeID num_active_nodes,
      const ClusterID num_clusters,
      const ClusterID,
      const Options<NodeID, ClusterID> &options
  ) {
    if (options.active_set_strategy == ActiveSetStrategy::LOCAL) {
      if (active.size() < num_nodes) {
        active.resize(num_nodes);
      }
    } else if (options.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      if (active.size() < num_active_nodes) {
        active.resize(num_active_nodes);
      }
    }

    if (options.use_two_hop_clustering) {
      if (favored_clusters.size() < num_active_nodes) {
        favored_clusters.resize(num_active_nodes);
      }
    }

    if (rating_map_ets.empty() || _rating_map_capacity < num_clusters) {
      rating_map_ets = tbb::enumerable_thread_specific<RatingMap>([num_clusters] {
        return RatingMap(num_clusters);
      });
      _rating_map_capacity = num_clusters;
    } else {
      for (auto &rating_map : rating_map_ets) {
        rating_map.change_max_size(num_clusters);
      }
    }

    if (local_cluster_selection_states.size() <
        static_cast<std::size_t>(tbb::this_task_arena::max_concurrency())) {
      local_cluster_selection_states.resize(tbb::this_task_arena::max_concurrency());
    }
  }

  void free() {
    rating_map_ets.clear();
    growing_rating_map_ets.clear();
    tie_breaking_clusters_ets.clear();
    tie_breaking_favored_clusters_ets.clear();
    active.free();
    moved.free();
    favored_clusters.free();
    second_phase_nodes.clear();
    second_phase_nodes.shrink_to_fit();
    local_cluster_selection_states.clear();
    concurrent_rating_map.free();
    _rating_map_capacity = 0;
  }

private:
  ClusterID _rating_map_capacity = 0;
};

template <
    typename Graph,
    typename LabelStore,
    typename WeightStore,
    typename ClusterSelector,
    typename NeighborPolicy,
    typename Workspace>
class LabelPropagationCore {
  SET_STATISTICS_FROM_GLOBAL();

public:
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename LabelStore::ClusterIDType;
  using ClusterWeight = typename WeightStore::ClusterWeightType;
  using RatingMap = typename Workspace::RatingMapType;
  using GrowingRatingMap = typename Workspace::GrowingRatingMapType;
  using ConcurrentRatingMap = typename Workspace::ConcurrentRatingMapType;
  using SelectionState = ClusterSelectionState<NodeID, ClusterID, ClusterWeight, EdgeWeight>;
  using Result = PassResult<NodeID, ClusterID, EdgeWeight>;

  struct PassStats {
    NodeID processed_nodes = 0;
    NodeID moved_nodes = 0;
    ClusterID removed_clusters = 0;
    EdgeWeight expected_total_gain = 0;
  };

  struct BestMove {
    NodeID node;
    NodeWeight node_weight;
    ClusterID old_cluster;
    ClusterID new_cluster;
    EdgeWeight gain;
    bool valid = false;
  };

  LabelPropagationCore(
      const Graph &graph,
      LabelStore &labels,
      WeightStore &weights,
      ClusterSelector &selector,
      NeighborPolicy &neighbors,
      Workspace &workspace,
      Options<NodeID, ClusterID> options
  )
      : _graph(graph),
        _labels(labels),
        _weights(weights),
        _selector(selector),
        _neighbors(neighbors),
        _workspace(workspace),
        _options(options) {}

  void set_options(const Options<NodeID, ClusterID> options) {
    _options = options;
  }

  [[nodiscard]] const Options<NodeID, ClusterID> &options() const {
    return _options;
  }

  void initialize(const Initialization<NodeID, ClusterID> init) {
    _num_nodes = init.num_nodes;
    _num_active_nodes = init.num_active_nodes;
    _prev_num_clusters = _num_clusters;
    _num_clusters = init.num_clusters;
    _initial_num_clusters = init.num_clusters;
    _current_num_clusters = init.num_clusters;
    _relabeled = false;
    _workspace.allocate(_num_nodes, _num_active_nodes, _num_clusters, _prev_num_clusters, _options);
    reset_state();
  }

  void clear_iteration_order_cache() {}

  [[nodiscard]] ClusterID current_num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] bool should_stop() const {
    return _options.track_cluster_count && _current_num_clusters <= _options.desired_num_clusters;
  }

  class Pass {
  public:
    explicit Pass(LabelPropagationCore &core) : _core(core) {}

    class Local {
    public:
      Local(LabelPropagationCore &core, PassStats &stats) : _core(core), _stats(stats) {}

      [[nodiscard]] bool should_consider(const NodeID u) const {
        return _core.should_consider(u);
      }

      [[nodiscard]] BestMove find_best_move(const NodeID u) {
        return _core.find_best_move(u, Random::instance());
      }

      std::pair<bool, bool> try_commit_move(const BestMove &move) {
        return _core.try_commit_move(move, _stats);
      }

      void handle_next_node(const NodeID u) {
        _current_node = u;
        if (!should_consider(u)) {
          return;
        }

        ++_stats.processed_nodes;

        switch (_core._options.rating_map_strategy) {
        case RatingMapStrategy::GROWING_HASH_TABLES:
          handle_node_with_map(_core._workspace.growing_rating_map_ets.local());
          break;

        case RatingMapStrategy::SINGLE_PHASE:
          handle_node_with_map(_core._workspace.rating_map_ets.local());
          break;

        case RatingMapStrategy::TWO_PHASE:
          handle_first_phase_node();
          break;
        }
      }

    private:
      template <typename LocalRatingMap> void handle_node_with_map(LocalRatingMap &rating_map) {
        auto &rand = Random::instance();
        auto &tie_breaking_clusters = _core._workspace.tie_breaking_clusters_ets.local();
        auto &tie_breaking_favored_clusters =
            _core._workspace.tie_breaking_favored_clusters_ets.local();

        const BestMove move = _core.find_best_move(
            _current_node, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
        );
        _core.try_commit_move(move, _stats);
      }

      void handle_first_phase_node() {
        auto &rand = Random::instance();
        auto &rating_map = _core._workspace.rating_map_ets.local();
        auto &tie_breaking_clusters = _core._workspace.tie_breaking_clusters_ets.local();
        auto &tie_breaking_favored_clusters =
            _core._workspace.tie_breaking_favored_clusters_ets.local();

        if constexpr (Workspace::kSupportsTwoPhase) {
          const NodeID u = _current_node;
          const NodeWeight u_weight = _core._graph.node_weight(u);
          const ClusterID u_cluster = _core._labels.cluster(u);
          const std::size_t upper_bound_size = std::min<std::size_t>(
              {_core._graph.degree(u),
               _core._initial_num_clusters,
               _core._options.rating_map_threshold}
          );

          const auto maybe_move = rating_map.execute(upper_bound_size, [&](auto &actual_map) {
            return _core.find_best_move_first_phase(
                u,
                u_weight,
                u_cluster,
                rand,
                actual_map,
                tie_breaking_clusters,
                tie_breaking_favored_clusters
            );
          });

          if (maybe_move.has_value()) {
            _core.try_commit_move(*maybe_move, _stats);
          }
        } else {
          KASSERT(false, "two-phase label propagation is not supported by this workspace");
        }
      }

      friend class Pass;

      LabelPropagationCore &_core;
      PassStats &_stats;
      NodeID _current_node = 0;
    };

    [[nodiscard]] Local local() {
      return Local(_core, _stats.local());
    }

    void handle_next_node(const NodeID u) {
      auto local_pass = local();
      local_pass._current_node = u;
      local_pass.handle_next_node(u);
    }

    [[nodiscard]] Result finish() {
      if (_core._options.rating_map_strategy == RatingMapStrategy::TWO_PHASE) {
        _core.finish_second_phase(_stats.local());
      }

      Result result;
      for (const PassStats &local_stats : _stats) {
        result.processed_nodes += local_stats.processed_nodes;
        result.moved_nodes += local_stats.moved_nodes;
        result.removed_clusters += local_stats.removed_clusters;
        result.expected_total_gain += local_stats.expected_total_gain;
      }

      _core._current_num_clusters -= result.removed_clusters;
      _core._expected_total_gain += result.expected_total_gain;
      return result;
    }

  private:
    LabelPropagationCore &_core;
    tbb::enumerable_thread_specific<PassStats> _stats;
  };

  [[nodiscard]] Pass begin_pass() {
    _workspace.second_phase_nodes.clear();
    return Pass(*this);
  }

  template <typename LocalRatingMap>
  [[nodiscard]] BestMove find_best_move(
      const NodeID u,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const NodeWeight u_weight = _graph.node_weight(u);
    const ClusterID u_cluster = _labels.cluster(u);
    return find_best_move_impl(
        u,
        u_weight,
        u_cluster,
        rand,
        rating_map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );
  }

  [[nodiscard]] BestMove find_best_move(const NodeID u, Random &rand) {
    auto &rating_map = _workspace.rating_map_ets.local();
    auto &tie_breaking_clusters = _workspace.tie_breaking_clusters_ets.local();
    auto &tie_breaking_favored_clusters = _workspace.tie_breaking_favored_clusters_ets.local();
    return find_best_move(
        u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  std::pair<bool, bool> try_commit_move(const BestMove &move, PassStats &stats) {
    if (!move.valid || _labels.cluster(move.node) == move.new_cluster) {
      return {false, false};
    }

    const bool successful_weight_move = _weights.move_cluster_weight(
        move.old_cluster,
        move.new_cluster,
        move.node_weight,
        _weights.max_cluster_weight(move.new_cluster)
    );

    if (!successful_weight_move) {
      return {false, false};
    }

    _labels.move_node(move.node, move.new_cluster);
    activate_neighbors(move.node);
    stats.expected_total_gain += move.gain;

    const bool emptied_cluster =
        _options.track_cluster_count && _weights.cluster_weight(move.old_cluster) == 0;
    if (emptied_cluster) {
      ++stats.removed_clusters;
    }
    ++stats.moved_nodes;
    return {true, emptied_cluster};
  }

  void activate_neighbors(const NodeID u) {
    if (_options.active_set_strategy == ActiveSetStrategy::NONE) {
      return;
    }

    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active.size()) {
        __atomic_store_n(&_workspace.active[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

  void activate_neighbors_of_ghost_node(const NodeID u) {
    KASSERT(_graph.is_ghost_node(u));
    if (_options.active_set_strategy != ActiveSetStrategy::GLOBAL) {
      return;
    }

    _graph.ghost_graph().adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active.size()) {
        __atomic_store_n(&_workspace.active[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

  void match_isolated_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    handle_isolated_nodes<true>(from, to);
  }

  void cluster_isolated_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    handle_isolated_nodes<false>(from, to);
  }

  void
  match_two_hop_nodes(const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()) {
    handle_two_hop_nodes<true, false>(from, to);
  }

  void cluster_two_hop_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    handle_two_hop_nodes<false, false>(from, to);
  }

  void match_two_hop_nodes_threadwise(
      const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    handle_two_hop_nodes<true, true>(from, to);
  }

  void cluster_two_hop_nodes_threadwise(
      const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    handle_two_hop_nodes<false, true>(from, to);
  }

  void relabel_clusters() {
    SCOPED_HEAP_PROFILER("Relabel");
    SCOPED_TIMER("Relabel");

    ClusterID num_actual_clusters = _current_num_clusters;
    _initial_num_clusters = num_actual_clusters;
    _relabeled = true;

    if (_workspace.moved.size() < _graph.n()) {
      _workspace.moved.resize(_graph.n());
    }

    StaticArray<ClusterID> mapping(_graph.n());
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph.n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const ClusterID c_u = _labels.cluster(u);
        __atomic_store_n(&mapping[c_u], 1, __ATOMIC_RELAXED);

        if (u != c_u) {
          _workspace.moved[u] = 1;
        }
      }
    });

    parallel::prefix_sum(mapping.begin(), mapping.end(), mapping.begin());
    KASSERT(num_actual_clusters == mapping[_graph.n() - 1]);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph.n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        _labels.move_node(u, mapping[_labels.cluster(u)] - 1);

        if (u < _workspace.favored_clusters.size()) {
          _workspace.favored_clusters[u] = mapping[_workspace.favored_clusters[u]] - 1;
        }
      }
    });

    _weights.reassign_cluster_weights(mapping, num_actual_clusters);
  }

private:
  [[nodiscard]] bool should_consider(const NodeID u) const {
    if (u >= _num_active_nodes) {
      return false;
    }
    if (_neighbors.skip(u)) {
      return false;
    }
    if (_graph.degree(u) >= _options.max_degree) {
      return false;
    }
    if (_options.active_set_strategy != ActiveSetStrategy::NONE) {
      if (u >= _workspace.active.size() ||
          !__atomic_load_n(&_workspace.active[u], __ATOMIC_RELAXED)) {
        return false;
      }
    }
    return true;
  }

  template <typename LocalRatingMap>
  [[nodiscard]] BestMove find_best_move_impl(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    auto action = [&](auto &map) {
      return compute_best_move_from_map(
          u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    };

    if constexpr (std::is_same_v<LocalRatingMap, GrowingRatingMap>) {
      return action(rating_map);
    } else {
      const std::size_t upper_bound_size =
          std::min<ClusterID>(_graph.degree(u), _initial_num_clusters);
      return rating_map.execute(upper_bound_size, action);
    }
  }

  template <typename LocalRatingMap>
  [[nodiscard]] std::optional<BestMove> find_best_move_first_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_second_phase_node = false;
    bool is_interface_node = false;

    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) -> bool {
      if (_neighbors.accept(u, v)) {
        const ClusterID v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (map.size() >= _options.rating_map_threshold) [[unlikely]] {
          is_second_phase_node = true;
          return true;
        }

        if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }

      return false;
    };

    if (_options.max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _options.max_num_neighbors, add_to_rating_map);
    }

    if (is_second_phase_node) [[unlikely]] {
      map.clear();
      _workspace.second_phase_nodes.push_back(u);
      return std::nullopt;
    }

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <typename ActualMap>
  [[nodiscard]] BestMove compute_best_move_from_map(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_interface_node = false;

    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) {
      if (_neighbors.accept(u, v)) {
        const ClusterID v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }
    };

    if (_options.max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _options.max_num_neighbors, add_to_rating_map);
    }

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <typename ActualMap>
  [[nodiscard]] BestMove compute_best_move_after_rating(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);
    SelectionState state{
        .local_rand = rand,
        .u = u,
        .u_weight = u_weight,
        .initial_cluster = u_cluster,
        .initial_cluster_weight = initial_cluster_weight,
        .best_cluster = u_cluster,
        .best_gain = 0,
        .best_cluster_weight = initial_cluster_weight,
        .overall_best_gain = 0,
        .current_cluster = 0,
        .current_gain = 0,
        .current_cluster_weight = 0,
    };

    const bool store_favored_cluster =
        _options.use_two_hop_clustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _options.use_actual_gain ? map[u_cluster] : 0;
    const ClusterID favored_cluster = _selector.select(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
      _workspace.favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = state.best_gain - map[state.initial_cluster];
    map.clear();
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = state.best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  void clear_active(const NodeID u, const bool is_interface_node) {
    if (_options.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      __atomic_store_n(&_workspace.active[u], 0, __ATOMIC_RELAXED);
    } else if (_options.active_set_strategy == ActiveSetStrategy::LOCAL && !is_interface_node) {
      __atomic_store_n(&_workspace.active[u], 0, __ATOMIC_RELAXED);
    }
  }

  void finish_second_phase(PassStats &stats) {
    if constexpr (Workspace::kSupportsTwoPhase) {
      const std::size_t num_clusters = _initial_num_clusters;
      if (_workspace.concurrent_rating_map.capacity() < num_clusters) {
        _workspace.concurrent_rating_map.resize(num_clusters);
      }

      if (!_workspace.second_phase_nodes.empty() && _options.relabel_before_second_phase) {
        relabel_clusters();
      }

      auto &rand = Random::instance();
      for (const NodeID u : _workspace.second_phase_nodes) {
        if (_neighbors.skip(u)) {
          continue;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        const ClusterID u_cluster = _labels.cluster(u);
        const BestMove move = find_best_move_second_phase(
            u, u_weight, u_cluster, rand, _workspace.concurrent_rating_map
        );
        const auto [moved, emptied] = try_commit_move(move, stats);

        if (moved && _relabeled && u < _workspace.moved.size()) {
          _workspace.moved[u] = 1;
        }
        (void)emptied;
      }

      _workspace.second_phase_nodes.clear();
    } else {
      KASSERT(false, "two-phase label propagation is not supported by this workspace");
    }
  }

  [[nodiscard]] BestMove find_best_move_second_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ConcurrentRatingMap &map
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);

    const auto flush_local_rating_map = [&](auto &local_used_entries, auto &local_rating_map) {
      for (const auto [cluster, rating] : local_rating_map.entries()) {
        const EdgeWeight prev_rating = __atomic_fetch_add(&map[cluster], rating, __ATOMIC_RELAXED);

        if (prev_rating == 0) {
          local_used_entries.push_back(cluster);
        }
      }

      local_rating_map.clear();
    };

    bool is_interface_node = false;
    _graph.pfor_adjacent_nodes(
        u, _options.max_num_neighbors, 2000, [&](auto &&pfor_adjacent_nodes) {
          auto &local_used_entries = map.local_used_entries();
          auto &local_rating_map = _workspace.rating_map_ets.local().small_map();

          pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
            if (_neighbors.accept(u, v)) {
              const ClusterID v_cluster = _labels.cluster(v);
              local_rating_map[v_cluster] += w;

              if (local_rating_map.size() >= _options.rating_map_threshold) [[unlikely]] {
                flush_local_rating_map(local_used_entries, local_rating_map);
              }

              if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
                is_interface_node |= v >= _num_active_nodes;
              }
            }
          });
        }
    );

    tbb::parallel_for(_workspace.rating_map_ets.range(), [&](auto &rating_maps) {
      auto &local_used_entries = map.local_used_entries();
      for (auto &rating_map : rating_maps) {
        auto &local_rating_map = rating_map.small_map();
        flush_local_rating_map(local_used_entries, local_rating_map);
      }
    });

    clear_active(u, is_interface_node);

    const bool store_favored_cluster =
        _options.use_two_hop_clustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _options.use_actual_gain ? map[u_cluster] : 0;

    map.iterate_and_reset([&](const auto i, auto &local_entries) {
      SelectionState local_state{
          .local_rand = Random::instance(),
          .u = u,
          .u_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .best_cluster = u_cluster,
          .best_gain = 0,
          .best_cluster_weight = initial_cluster_weight,
          .overall_best_gain = 0,
          .current_cluster = 0,
          .current_gain = 0,
          .current_cluster_weight = 0,
      };

      const ClusterID local_favored_cluster = _selector.select(
          store_favored_cluster,
          gain_delta,
          local_state,
          local_entries,
          _workspace.tie_breaking_clusters_ets.local(),
          _workspace.tie_breaking_favored_clusters_ets.local()
      );
      const EdgeWeight local_favored_cluster_gain = map[local_favored_cluster];

      _workspace.local_cluster_selection_states[i] = {
          local_state.best_gain,
          local_state.best_cluster,
          local_favored_cluster_gain,
          local_favored_cluster,
      };
    });

    ClusterID favored_cluster = u_cluster;
    ClusterID best_cluster = u_cluster;
    EdgeWeight best_gain = 0;

    if (_options.tie_breaking_strategy == TieBreakingStrategy::UNIFORM) {
      auto &tie_breaking_clusters = _workspace.tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _workspace.tie_breaking_favored_clusters_ets.local();

      EdgeWeight favored_cluster_gain = 0;
      for (auto &local_state : _workspace.local_cluster_selection_states) {
        if (local_state.best_gain > best_gain) {
          best_gain = local_state.best_gain;
          best_cluster = local_state.best_cluster;

          tie_breaking_clusters.clear();
          tie_breaking_clusters.push_back(local_state.best_cluster);
        } else if (local_state.best_gain == best_gain) {
          tie_breaking_clusters.push_back(local_state.best_cluster);
        }

        if (store_favored_cluster) {
          if (local_state.favored_cluster_gain > favored_cluster_gain) {
            favored_cluster_gain = local_state.favored_cluster_gain;
            favored_cluster = local_state.favored_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(local_state.favored_cluster);
          } else if (local_state.favored_cluster_gain == favored_cluster_gain) {
            tie_breaking_favored_clusters.push_back(local_state.favored_cluster);
          }
        }

        local_state.best_gain = -1;
        local_state.favored_cluster_gain = -1;
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = rand.random_index(0, tie_breaking_clusters.size());
        best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();
    } else {
      EdgeWeight favored_cluster_gain = 0;
      for (auto &local_state : _workspace.local_cluster_selection_states) {
        if (local_state.best_gain > best_gain) {
          best_gain = local_state.best_gain;
          best_cluster = local_state.best_cluster;
        }

        if (store_favored_cluster && local_state.favored_cluster_gain > favored_cluster_gain) {
          favored_cluster_gain = local_state.favored_cluster_gain;
          favored_cluster = local_state.favored_cluster;
        }

        local_state.best_gain = -1;
        local_state.favored_cluster_gain = -1;
      }
    }

    if (store_favored_cluster && best_cluster == u_cluster) {
      _workspace.favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = best_gain - map[u_cluster];
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for<NodeID>(0, _num_active_nodes, [&](const NodeID u) {
            if (_options.active_set_strategy != ActiveSetStrategy::NONE) {
              _workspace.active[u] = 1;
            }

            const ClusterID initial_cluster = _labels.initial_cluster(u);
            _labels.init_cluster(u, initial_cluster);
            if (_options.use_two_hop_clustering && u < _workspace.favored_clusters.size()) {
              _workspace.favored_clusters[u] = initial_cluster;
            }
            if (u < _workspace.moved.size()) {
              _workspace.moved[u] = 0;
            }
            if constexpr (requires(LabelStore labels, NodeID node) {
                            labels.reset_node_state(node);
                          }) {
              _labels.reset_node_state(u);
            }
          });
        },
        [&] {
          tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
            _weights.init_cluster_weight(cluster, _weights.initial_cluster_weight(cluster));
          });
        }
    );
    _expected_total_gain = 0;
    _current_num_clusters = _initial_num_clusters;
  }

  template <bool kMatch> void handle_isolated_nodes(const NodeID from, const NodeID to) {
    constexpr ClusterID kInvalidClusterID = std::numeric_limits<ClusterID>::max();
    tbb::enumerable_thread_specific<ClusterID> current_cluster_ets(kInvalidClusterID);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(_graph.n(), to)),
        [&](tbb::blocked_range<NodeID> r) {
          ClusterID cluster = current_cluster_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (_graph.degree(u) == 0) {
              const ClusterID cu = _labels.cluster(u);

              if (cluster != kInvalidClusterID &&
                  _weights.move_cluster_weight(
                      cu, cluster, _weights.cluster_weight(cu), _weights.max_cluster_weight(cluster)
                  )) {
                _labels.move_node(u, cluster);
                if constexpr (kMatch) {
                  cluster = kInvalidClusterID;
                }
              } else {
                cluster = cu;
              }
            }
          }

          current_cluster_ets.local() = cluster;
        }
    );
  }

  template <bool kMatch, bool kThreadwise>
  void handle_two_hop_nodes(const NodeID from, const NodeID to) {
    KASSERT(_options.use_two_hop_clustering);

    if constexpr (kThreadwise) {
      handle_two_hop_nodes_threadwise_impl<kMatch>(from, to);
    } else {
      handle_two_hop_nodes_impl<kMatch>(from, to);
    }
  }

  [[nodiscard]] bool is_considered_for_two_hop_clustering(const NodeID u) {
    if (_graph.degree(u) == 0) {
      return false;
    }

    const auto check_cluster_weight = [&](const ClusterID c_u) {
      const ClusterWeight current_weight = _weights.cluster_weight(c_u);

      if (current_weight > _weights.max_cluster_weight(c_u) / 2 ||
          current_weight != _weights.initial_cluster_weight(c_u)) {
        return false;
      }

      return true;
    };

    if (_relabeled) {
      if (u < _workspace.moved.size() && _workspace.moved[u]) {
        return false;
      }

      const ClusterID c_u = _labels.cluster(u);
      return check_cluster_weight(c_u);
    } else {
      if (u != _labels.cluster(u)) {
        return false;
      }

      return check_cluster_weight(u);
    }
  }

  template <bool kMatch>
  void handle_two_hop_nodes_threadwise_impl(const NodeID from, const NodeID to) {
    tbb::enumerable_thread_specific<DynamicFlatMap<ClusterID, NodeID>> matching_map_ets;

    auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
      const ClusterID c_u = _labels.cluster(u);
      ClusterID &rep_key = matching_map[_workspace.favored_clusters[u]];

      if (rep_key == 0) {
        rep_key = c_u + 1;
      } else {
        const ClusterID rep = rep_key - 1;

        const bool could_move_u_to_rep = _weights.move_cluster_weight(
            c_u, rep, _weights.cluster_weight(c_u), _weights.max_cluster_weight(rep)
        );

        if constexpr (kMatch) {
          KASSERT(could_move_u_to_rep);
          _labels.move_node(u, rep);
          rep_key = 0;
        } else {
          if (could_move_u_to_rep) {
            _labels.move_node(u, rep);
          } else {
            rep_key = c_u + 1;
          }
        }
      }
    };

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(to, _graph.n()), 512),
        [&](const tbb::blocked_range<NodeID> &r) {
          auto &matching_map = matching_map_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (is_considered_for_two_hop_clustering(u)) {
              handle_node(matching_map, u);
            }
          }
        }
    );
  }

  template <bool kMatch> void handle_two_hop_nodes_impl(const NodeID from, const NodeID to) {
    const auto is_considered_for_non_threadwise_two_hop_clustering = [&](const NodeID u) {
      if (_graph.degree(u) == 0 || u != _labels.cluster(u)) {
        return false;
      }

      const ClusterWeight current_weight = _weights.cluster_weight(u);
      return current_weight <= _weights.max_cluster_weight(u) / 2 &&
             current_weight == _weights.initial_cluster_weight(u);
    };

    tbb::parallel_for(from, std::min(to, _graph.n()), [&](const NodeID u) {
      if (is_considered_for_non_threadwise_two_hop_clustering(u)) {
        const NodeID cluster = _workspace.favored_clusters[u];
        if (is_considered_for_non_threadwise_two_hop_clustering(cluster) &&
            _weights.move_cluster_weight(
                u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
            )) {
          _labels.move_node(u, cluster);
          --_current_num_clusters;
        }
      } else {
        _workspace.favored_clusters[u] = u;
      }
    });

    tbb::parallel_for(from, std::min(to, _graph.n()), [&](const NodeID u) {
      if (should_stop() || !is_considered_for_non_threadwise_two_hop_clustering(u)) {
        return;
      }

      const NodeID C = __atomic_load_n(&_workspace.favored_clusters[u], __ATOMIC_RELAXED);
      auto &sync = _workspace.favored_clusters[C];

      do {
        NodeID cluster = sync;

        if (cluster == C) {
          if (__atomic_compare_exchange_n(
                  &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )) {
            break;
          }
          if (cluster == C) {
            continue;
          }
        }

        KASSERT(__atomic_load_n(&_workspace.favored_clusters[cluster], __ATOMIC_RELAXED) == C);

        if constexpr (kMatch) {
          if (__atomic_compare_exchange_n(
                  &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )) {
            [[maybe_unused]] const bool success = _weights.move_cluster_weight(
                u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
            );
            KASSERT(success);

            _labels.move_node(u, cluster);
            break;
          }
        } else {
          if (_weights.move_cluster_weight(
                  u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
              )) {
            _labels.move_node(u, cluster);
            break;
          } else if (
              __atomic_compare_exchange_n(
                  &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )
          ) {
            break;
          }
        }
      } while (true);
    });
  }

  const Graph &_graph;
  LabelStore &_labels;
  WeightStore &_weights;
  ClusterSelector &_selector;
  NeighborPolicy &_neighbors;
  Workspace &_workspace;
  Options<NodeID, ClusterID> _options;

  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
  ClusterID _initial_num_clusters = 0;
  parallel::Atomic<ClusterID> _current_num_clusters = 0;
  parallel::Atomic<EdgeWeight> _expected_total_gain = 0;
  bool _relabeled = false;
};

template <typename Order, typename Core>
typename Core::Result run_iteration(Order &order, Core &lp) {
  auto pass = lp.begin_pass();
  order.parallel_for_each([&](const auto node) {
    auto local = pass.local();
    local.handle_next_node(node);
  });
  return pass.finish();
}

} // namespace kaminpar::lp
