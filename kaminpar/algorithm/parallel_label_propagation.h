/*******************************************************************************
 * @file:   parallel_label_propagation.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Generic implementation of label propagation.
 ******************************************************************************/
#pragma once

#include "datastructure/rating_map.h"
#include "parallel.h"
#include "utility/random.h"
#include "utility/timer.h"

#include <atomic>
#include <ranges>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/scalable_allocator.h>
#include <type_traits>

namespace kaminpar {
struct LabelPropagationConfig {
  using Graph = ::kaminpar::Graph;

  // Data structure used to accumulate edge weights for gain value calculation
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, FastResetArray<EdgeWeight>>;

  // Data type for cluster IDs and weights
  using ClusterID = Mandatory;
  using ClusterWeight = Mandatory;

  // Approx. number of edges per work unit
  static constexpr NodeID kMinChunkSize = 1024;

  // Nodes per permutation unit: when iterating over nodes in a chunk, we divide them into permutation units, iterate
  // over permutation orders in random order, and iterate over nodes inside a permutation unit in random order.
  static constexpr NodeID kPermutationSize = 64;

  // When randomizing the node order inside a permutation unit, we pick a random permutation from a pool of
  // permutations. This constant determines the pool size.
  static constexpr std::size_t kNumberOfNodePermutations = 64;

  // If true, we count the number of empty clusters
  static constexpr bool kTrackClusterCount = false;

  // If true, try to shrink the number of clusters by joining clusters with distance 2
  static constexpr bool kUseTwoHopClustering = false;
};

/**
 * Generic implementation of parallelized label propagation.
 */
template<typename Derived, std::derived_from<LabelPropagationConfig> Config>
class LabelPropagation {
  SET_DEBUG(false);
  SET_STATISTICS(false);

  using Self = LabelPropagation<Derived, Config>;

protected:
  using RatingMap = typename Config::RatingMap;
  using Graph = typename Config::Graph;
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename Config::ClusterID;
  using ClusterWeight = typename Config::ClusterWeight;

public:
  void set_max_degree(const NodeID max_degree) { _max_degree = max_degree; }
  [[nodiscard]] NodeID max_degree() const { return _max_degree; }

  void set_max_num_neighbors(const ClusterID max_num_neighbors) { _max_num_neighbors = max_num_neighbors; }
  [[nodiscard]] ClusterID max_num_neighbors() const { return _max_num_neighbors; }

  void set_desired_num_clusters(const ClusterID desired_num_clusters) { _desired_num_clusters = desired_num_clusters; }
  [[nodiscard]] ClusterID desired_num_clusters() const { return _desired_num_clusters; }

  [[nodiscard]] EdgeWeight expected_total_gain() const { return _expected_total_gain; }

protected:
  struct ClusterSelectionState {
    Randomize &local_rand;
    NodeID u;
    NodeWeight u_weight;
    ClusterID initial_cluster;
    ClusterWeight initial_cluster_weight;
    ClusterID best_cluster;
    EdgeWeight best_gain;
    ClusterWeight best_cluster_weight;
    ClusterID current_cluster;
    EdgeWeight current_gain;
    ClusterWeight current_cluster_weight;
  };

  explicit LabelPropagation(const ClusterID max_num_nodes)
      : _max_num_nodes{max_num_nodes},
        _active(max_num_nodes),
        _favored_clusters(Config::kUseTwoHopClustering * (max_num_nodes + 1)) {}

  void initialize(const Graph *graph, const ClusterID num_clusters) {
    _graph = graph;
    _initial_num_clusters = num_clusters;
    _current_num_clusters = num_clusters;
    reset_state();
  }

  bool should_stop() {
    if (Config::kTrackClusterCount) { return _current_num_clusters <= _desired_num_clusters; }
    return false;
  }

  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for(static_cast<ClusterID>(0), _graph->n(), [&](const auto u) {
            _active[u] = 1;
            derived_init_cluster(u, derived_initial_cluster(u));
            if constexpr (Config::kUseTwoHopClustering) { _favored_clusters[u] = derived_initial_cluster(u); }
            derived_reset_node_state(u);
          });
        },
        [&] {
          tbb::parallel_for(static_cast<ClusterID>(0), _initial_num_clusters, [&](const auto cluster) {
            derived_init_cluster_weight(cluster, derived_initial_cluster_weight(cluster));
          });
        });
    IFSTATS(_expected_total_gain = 0);
    _current_num_clusters = _initial_num_clusters;
  }

  // returns the following status flags:
  // ... first:  whether the node could be moved to another cluster
  // ... second: whether the previous cluster of the node is now empty (only if Config::kReportEmptyClusters)
  std::pair<bool, bool> handle_node(const NodeID u, Randomize &local_rand, auto &local_rating_map) {
    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived_cluster(u);
    const auto [new_cluster, new_gain] = find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    if (derived_cluster(u) != new_cluster) {
      if (derived_move_cluster_weight(u_cluster, new_cluster, u_weight, derived_max_cluster_weight(new_cluster))) {
        derived_move_node(u, new_cluster);
        activate_neighbors(u);
        IFSTATS(_expected_total_gain += new_gain);

        const bool decrement_cluster_count = Config::kTrackClusterCount && derived_cluster_weight(u_cluster) == 0;
        // do not update _current_num_clusters here to avoid fetch_add()
        return {true, decrement_cluster_count}; // did move, did reduce nonempty cluster count?
      }
    }

    // did not move, did not reduce cluster count
    return {false, false};
  }

  std::pair<ClusterID, EdgeWeight> find_best_cluster(const NodeID u, const NodeWeight u_weight,
                                                     const ClusterID u_cluster, Randomize &local_rand,
                                                     auto &local_rating_map) {
    auto action = [&](auto &map) {
      const ClusterWeight initial_cluster_weight = derived_cluster_weight(u_cluster);
      ClusterSelectionState state{
          .local_rand = local_rand,
          .u = u,
          .u_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .best_cluster = u_cluster,
          .best_gain = 0,
          .best_cluster_weight = initial_cluster_weight,
          .current_cluster = 0,
          .current_gain = 0,
          .current_cluster_weight = 0,
      };

      auto add_to_rating_map = [&](const EdgeID e, const NodeID v) {
        if (derived_accept_neighbor(v)) {
          const ClusterID v_cluster = derived_cluster(v);
          const EdgeWeight rating = _graph->edge_weight(e);
          map[v_cluster] += rating;
        }
      };

      const EdgeID from = _graph->first_edge(u);
      const EdgeID to = from + std::min(_graph->degree(u), _max_num_neighbors);
      for (EdgeID e = from; e < to; ++e) { add_to_rating_map(e, _graph->edge_target(e)); }

      // after LP, we might want to use 2-hop clustering to merge nodes that could not find any cluster to join
      // for this, we store a favored cluster for each node u if:
      // (1) we actually use 2-hop clustering
      // (2) u is still in a singleton cluster (weight of node == weight of cluster)
      // (3) the cluster is light (at most half full)
      ClusterID favored_cluster = u_cluster;
      const bool store_favored_cluster = Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
                                         initial_cluster_weight <= derived_max_cluster_weight(u_cluster) / 2;

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating;
        state.current_cluster_weight = derived_cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.best_gain) { favored_cluster = state.current_cluster; }

        if (derived_accept_cluster(state)) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      // if we couldn't join any cluster, we store the favored cluster
      if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
        _favored_clusters[u] = favored_cluster;
      }

      const EdgeWeight actual_gain = IFSTATS(state.best_gain - map[state.initial_cluster]);
      map.clear();
      return std::make_pair(state.best_cluster, actual_gain);
    };

    local_rating_map.update_upper_bound_size(_graph->degree(u));
    const auto [best_cluster, gain] = local_rating_map.run_with_map(action, action);

    return {best_cluster, gain};
  }

  void activate_neighbors(const NodeID u) {
    for (const NodeID v : _graph->adjacent_nodes(u)) {
      if (derived_accept_neighbor(v)) { _active[v].store(1, std::memory_order_relaxed); }
    }
  }

protected:
  //
  // 2-hop clustering
  //

  void perform_two_hop_clustering(const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()) {
    tbb::parallel_for(from, std::min<ClusterID>(to, _graph->n()), [&](const NodeID u) {
      // abort once we merged enough clusters
      if (should_stop()) { return; }

      // only consider nodes that
      // (a) are still alone in their cluster (== they still have their initial cluster weight)
      //     the first check, u != leader, prevents a cache miss on the cluster weight lookup for nodes that changed
      //     their block
      // (b) are light (== less than half full)
      const ClusterID leader = derived_cluster(u);
      if (u != leader) {
        const ClusterWeight initial_weight = derived_initial_cluster_weight(leader);
        const ClusterWeight current_weight = derived_cluster_weight(leader);
        const ClusterWeight max_weight = derived_max_cluster_weight(leader);
        if (current_weight != initial_weight || current_weight > max_weight / 2) { return; }
      }

      NodeID favored_leader = _favored_clusters[u];
      const bool u_is_isolated = (u == favored_leader);    // this only happens if u is an isolated node in the graph
      if (u_is_isolated) { favored_leader = _graph->n(); } // cluster isolated nodes together

      do {
        // we use _favored_clusters[l] to pair singleton clusters (nodes) that have favored cluster l
        // initially, this _favored_clusters[l] is set to l
        // the first node u replaces this value with u
        // the second node v sees that _favored_clusters[l] = u has a value other than l and joins the cluster of v
        // if possible

        // this works, since for the represented node `l` of a favored cluster of node `u`, the following assertions
        // hold:
        // (a) they are not in a light cluster, otherwise, u could have joined the cluster
        // (b) thus, they are not traversed in this function
        // (c) thus, they don't use their _favored_clusters[] entry, and it is _favored_clusters[l] = l since
        //     pick_cluster() does not change their entry

        NodeID expected_value = favored_leader;
        if (_favored_clusters[favored_leader].compare_exchange_strong(expected_value, u)) {
          break; // if this worked, we replaced favored_leader with u
        }

        // if this didn't work, there is another node that has the same favored leader -> try to join that nodes
        // cluster
        const NodeID partner = expected_value;
        if (_favored_clusters[favored_leader].compare_exchange_strong(expected_value, favored_leader)) {
          if (derived_move_cluster_weight(leader, partner, derived_cluster_weight(leader),
                                          derived_max_cluster_weight(partner))) {
            derived_move_node(u, partner);
            --_current_num_clusters;
          }
          break;
        }
      } while (true);
    });
  }

  //
  // Template methods for derived class
  //

  void derived_reset_node_state(const NodeID u) { static_cast<Derived *>(this)->reset_node_state(u); }

  [[nodiscard]] ClusterID derived_initial_cluster(const NodeID u) {
    return static_cast<Derived *>(this)->initial_cluster(u);
  }

  [[nodiscard]] ClusterID derived_cluster(const NodeID u) { return static_cast<Derived *>(this)->cluster(u); }

  void derived_init_cluster(const NodeID u, const ClusterID cluster) {
    static_cast<Derived *>(this)->init_cluster(u, cluster);
  }

  void derived_move_node(const NodeID u, const ClusterID cluster) {
    static_cast<Derived *>(this)->move_node(u, cluster);
  }

  [[nodiscard]] ClusterWeight derived_initial_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->initial_cluster_weight(cluster);
  }

  [[nodiscard]] ClusterWeight derived_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->cluster_weight(cluster);
  }

  void derived_init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    static_cast<Derived *>(this)->init_cluster_weight(cluster, weight);
  }

  [[nodiscard]] bool derived_move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster,
                                                 const ClusterWeight delta, const ClusterWeight max_weight) {
    return static_cast<Derived *>(this)->move_cluster_weight(old_cluster, new_cluster, delta, max_weight);
  }

  [[nodiscard]] ClusterWeight derived_max_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->max_cluster_weight(cluster);
  }

  [[nodiscard]] bool derived_accept_cluster(const ClusterSelectionState &state) {
    return static_cast<Derived *>(this)->accept_cluster(state);
  }

  [[nodiscard]] inline bool derived_accept_neighbor(const NodeID u) {
    return static_cast<Derived *>(this)->accept_neighbor(u);
  }

  //
  // Default implementation for optional template methods
  //

  void reset_node_state(const NodeID /* node */) {}

  [[nodiscard]] inline bool accept_neighbor(const NodeID /* node */) const { return true; }

protected:
  const ClusterID _max_num_nodes;

  const Graph *_graph{nullptr};
  ClusterID _initial_num_clusters;
  parallel::IntegralAtomicWrapper<ClusterID> _current_num_clusters;

  ClusterID _max_degree{std::numeric_limits<ClusterID>::max()};
  ClusterID _max_num_neighbors{std::numeric_limits<ClusterID>::max()};
  ClusterID _desired_num_clusters{0};

  tbb::enumerable_thread_specific<RatingMap> _rating_map_ets{[&] { return RatingMap{_max_num_nodes}; }};
  scalable_vector<parallel::IntegralAtomicWrapper<uint8_t>> _active;

  parallel::IntegralAtomicWrapper<EdgeWeight> _expected_total_gain;
  scalable_vector<parallel::IntegralAtomicWrapper<ClusterID>> _favored_clusters;
};

template<typename Derived, std::derived_from<LabelPropagationConfig> Config>
class InOrderLabelPropagation : public LabelPropagation<Derived, Config> {
  using Base = LabelPropagation<Derived, Config>;

protected:
  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;

  template<typename... Args>
  InOrderLabelPropagation(Args &&...args) : Base{std::forward<Args>(args)...} {}

  NodeID perform_iteration(const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()) {
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(from, std::min(_graph->n(), to)), [&](const auto &r) {
      EdgeID work_since_update = 0;
      NodeID num_removed_clusters = 0;

      auto &num_moved_nodes = num_moved_nodes_ets.local();
      auto &rand = Randomize::instance();
      auto &rating_map = _rating_map_ets.local();

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        if (work_since_update > Config::kMinChunkSize) {
          if (Base::should_stop()) { return; }

          _current_num_clusters -= num_removed_clusters;
          work_since_update = 0;
          num_removed_clusters = 0;
        }

        const auto [moved_node, emptied_cluster] = handle_node(u, rand, rating_map);
        work_since_update += _graph->degree(u);
        if (moved_node) { ++num_moved_nodes; }
        if (emptied_cluster) { ++num_removed_clusters; }
      }
    });

    return num_moved_nodes_ets.combine(std::plus{});
  }

  using Base::_current_num_clusters;
  using Base::_graph;
  using Base::_rating_map_ets;
};

template<typename Derived, std::derived_from<LabelPropagationConfig> Config>
class ChunkRandomizedLabelPropagation : public LabelPropagation<Derived, Config> {
  using Base = LabelPropagation<Derived, Config>;

protected:
  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;

  using Base::ClusterSelectionState;

  using Base::handle_node;
  using Base::set_max_degree;
  using Base::set_max_num_neighbors;
  using Base::should_stop;

  template<typename... Args>
  explicit ChunkRandomizedLabelPropagation(Args &&...args) : Base{std::forward<Args>(args)...} {}

  void initialize(const Graph *graph, const ClusterID num_clusters) {
    Base::initialize(graph, num_clusters);
    _chunks.clear();
    _buckets.clear();
  }

  NodeID perform_iteration(const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()) {
    ASSERT(from == 0 && to == std::numeric_limits<NodeID>::max())
        << "randomized iteration does not support node ranges";

    if (_chunks.empty()) { init_chunks(); }
    shuffle_chunks();

    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
    parallel::IntegralAtomicWrapper<std::size_t> next_chunk;

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (should_stop()) { return; }

      auto &local_num_moved_nodes = num_moved_nodes_ets.local();
      auto &local_rand = Randomize::instance();
      auto &local_rating_map = _rating_map_ets.local();
      NodeID num_removed_clusters = 0;

      const auto &chunk = _chunks[next_chunk.fetch_add(1, std::memory_order_relaxed)];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks = std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);
      std::vector<NodeID> sub_chunk_permutation(num_sub_chunks);
      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.end(), 0);
      local_rand.shuffle(sub_chunk_permutation);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start + Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u < chunk.end && _graph->degree(u) < _max_degree && _active[u].load(std::memory_order_relaxed)) {
            _active[u].store(0, std::memory_order_relaxed);

            const auto [moved_node, emptied_cluster] = handle_node(u, local_rand, local_rating_map);
            if (moved_node) { ++local_num_moved_nodes; }
            if (emptied_cluster) { ++num_removed_clusters; }
          }
        }
      }

      _current_num_clusters -= num_removed_clusters;
    });

    return num_moved_nodes_ets.combine(std::plus{});
  }

private:
  struct Chunk {
    NodeID start;
    NodeID end;
  };

  struct Bucket {
    std::size_t start;
    std::size_t end;
  };

  void shuffle_chunks() {
    tbb::parallel_for(static_cast<std::size_t>(0), _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Randomize::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  void init_chunks() {
    const auto max_bucket = std::min<std::size_t>(math::floor_log2(_max_degree), _graph->number_of_buckets());
    const EdgeID max_chunk_size = std::max<EdgeID>(Config::kMinChunkSize, std::sqrt(_graph->m()));
    const NodeID max_node_chunk_size = std::max<NodeID>(Config::kMinChunkSize, std::sqrt(_graph->n()));

    for (std::size_t bucket = 0; bucket < max_bucket; ++bucket) {
      const std::size_t bucket_size = _graph->bucket_size(bucket);
      if (bucket_size == 0) { continue; }

      parallel::IntegralAtomicWrapper<NodeID> offset = 0;
      tbb::enumerable_thread_specific<std::size_t> num_chunks_ets;
      tbb::enumerable_thread_specific<std::vector<Chunk>> chunks_ets;

      const std::size_t bucket_start = _graph->first_node_in_bucket(bucket);

      tbb::parallel_for(static_cast<int>(0), tbb::this_task_arena::max_concurrency(), [&](const int) {
        auto &chunks = chunks_ets.local();
        auto &num_chunks = num_chunks_ets.local();

        while (offset < bucket_size) {
          const NodeID begin = offset.fetch_add(max_node_chunk_size);
          if (begin >= bucket_size) { break; }
          const NodeID end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

          Degree current_chunk_size = 0;
          NodeID chunk_start = bucket_start + begin;

          for (NodeID i = begin; i < end; ++i) {
            const NodeID u = bucket_start + i;
            current_chunk_size += _graph->degree(u);
            if (current_chunk_size >= max_chunk_size) {
              chunks.emplace_back(chunk_start, u + 1);
              chunk_start = u + 1;
              current_chunk_size = 0;
              ++num_chunks;
            }
          }

          if (current_chunk_size > 0) {
            chunks.emplace_back(chunk_start, bucket_start + end);
            ++num_chunks;
          }
        }
      });

      const std::size_t num_chunks = num_chunks_ets.combine(std::plus{});

      const std::size_t chunks_start = _chunks.size();
      parallel::IntegralAtomicWrapper<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(chunks_ets.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
        }
      });

      _buckets.emplace_back(chunks_start, _chunks.size());
    }
  }

protected:
  using Base::_active;
  using Base::_current_num_clusters;
  using Base::_graph;
  using Base::_max_degree;
  using Base::_rating_map_ets;

  RandomPermutations<NodeID, Config::kPermutationSize, Config::kNumberOfNodePermutations> _random_permutations{};
  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;
};

template<typename NodeID, typename ClusterID>
class OwnedClusterVector {
protected:
  explicit OwnedClusterVector(const NodeID max_num_nodes) : _clusters(max_num_nodes) {}

  [[nodiscard]] auto &&take_clusters() { return std::move(_clusters); }

  [[nodiscard]] const auto &clusters() const { return _clusters; }

  void init_cluster(const NodeID node, const ClusterID cluster) { _clusters[node] = cluster; }

  [[nodiscard]] ClusterID cluster(const NodeID node) const { return _clusters[node]; }

  void move_node(const NodeID node, const ClusterID cluster) { _clusters[node] = cluster; }

private:
  scalable_vector<parallel::IntegralAtomicWrapper<ClusterID>> _clusters;
};

template<typename ClusterID, typename ClusterWeight>
class OwnedRelaxedClusterWeightVector {
protected:
  explicit OwnedRelaxedClusterWeightVector(const ClusterID max_num_clusters) : _cluster_weights(max_num_clusters) {}

  auto &&take_cluster_weights() { return std::move(_cluster_weights); }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) { _cluster_weights[cluster] = weight; }

  ClusterWeight cluster_weight(const ClusterID cluster) const { return _cluster_weights[cluster]; }

  bool move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
                           const ClusterWeight max_weight) {
    if (_cluster_weights[new_cluster] + delta <= max_weight) {
      _cluster_weights[new_cluster].fetch_add(delta, std::memory_order_relaxed);
      _cluster_weights[old_cluster].fetch_sub(delta, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

private:
  scalable_vector<parallel::IntegralAtomicWrapper<ClusterWeight>> _cluster_weights;
};
} // namespace kaminpar
