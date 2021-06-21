/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
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
/**
 * Generic implementation of parallelized label propagation.
 */
template<typename Derived, typename ClusterID, typename ClusterWeight>
class LabelPropagation {
  SET_DEBUG(false);
  SET_STATISTICS(false);

  // Approx. number of edges per work unit
  static constexpr NodeID kMinChunkSize = 1024;

  // Nodes per permutation unit: when iterating over nodes in a chunk, we divide them into permutation units, iterate
  // over permutation orders in random order, and iterate over nodes inside a permutation unit in random order.
  static constexpr NodeID kPermutationSize = 64;

  // We randomizing the node order inside a permutation unit, we pick a random permutation from a pool of permutations.
  // This constant determines the pool size.
  static constexpr std::size_t kNumberOfNodePermutations = 64;

public:
  void set_large_degree_threshold(const Degree large_degree_threshold) {
    _large_degree_threshold = large_degree_threshold;
  }

  void set_max_num_neighbors(const Degree max_num_neighbors) { _max_num_neighbors = max_num_neighbors; }

  [[nodiscard]] Degree large_degree_threshold() const { return _large_degree_threshold; }

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

  LabelPropagation(const NodeID max_n, const ClusterID max_num_clusters) : _max_n{max_n} {
    tbb::parallel_invoke([&] { _active.resize(_max_n); }, [&] { _cluster_weights.resize(max_num_clusters); });
  }

  void initialize(const Graph *graph) {
    _graph = graph;
    reset_state();
  }

  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for(static_cast<NodeID>(0), _graph->n(), [&](const NodeID u) {
            _active[u] = 1;
            derived_reset_node_state(u);
          });
        },
        [&] {
          tbb::parallel_for(static_cast<ClusterID>(0), derived_num_clusters(), [&](const ClusterID cluster) { //
            _cluster_weights[cluster] = derived_initial_cluster_weight(cluster);
          });
        });
    IFSTATS(_expected_total_gain = 0);

    _chunks.clear();
    _buckets.clear();
  }

  struct Chunk {
    NodeID start;
    NodeID end;
  };

  struct Bucket {
    std::size_t start;
    std::size_t end;
  };

  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;

  void shuffle_chunks() {
    tbb::parallel_for(static_cast<std::size_t>(0), _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Randomize::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  void init_chunks() {
    const auto max_bucket = std::min<std::size_t>(math::floor_log2(_large_degree_threshold),
                                                  _graph->number_of_buckets());
    const EdgeID max_chunk_size = std::max<EdgeID>(kMinChunkSize, std::sqrt(_graph->m()));
    const NodeID max_node_chunk_size = std::max<NodeID>(kMinChunkSize, std::sqrt(_graph->n()));

    for (std::size_t bucket = 0; bucket < max_bucket; ++bucket) {
      const std::size_t bucket_size = _graph->bucket_size(bucket);
      if (bucket_size == 0) { continue; }

      parallel::IntegralAtomicWrapper<NodeID> offset = 0;
      parallel::IntegralAtomicWrapper<std::size_t> num_chunks = 0;
      const std::size_t bucket_start = _graph->first_node_in_bucket(bucket);

      tbb::enumerable_thread_specific<std::vector<Chunk>> shared_chunks;

      tbb::parallel_for(static_cast<int>(0), tbb::this_task_arena::max_concurrency(), [&](const int) {
        auto &local_chunks = shared_chunks.local();

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
              local_chunks.emplace_back(chunk_start, u + 1);
              chunk_start = u + 1;
              current_chunk_size = 0;
              ++num_chunks;
            }
          }

          if (current_chunk_size > 0) {
            local_chunks.emplace_back(chunk_start, bucket_start + end);
            ++num_chunks;
          }
        }
      });

      const std::size_t chunks_start = _chunks.size();
      parallel::IntegralAtomicWrapper<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(shared_chunks.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
        }
      });

      _buckets.emplace_back(chunks_start, _chunks.size());
    }
  }

  std::pair<NodeID, NodeID> label_propagation_iteration() {
    ASSERT(_active.size() >= _graph->n());
    ASSERT(_cluster_weights.size() >= derived_num_clusters());

    if (_chunks.empty()) { init_chunks(); }
    shuffle_chunks();

    tbb::enumerable_thread_specific<NodeID> num_moved_nodes;
    parallel::IntegralAtomicWrapper<std::size_t> next_chunk;
    parallel::IntegralAtomicWrapper<NodeID> global_emptied_clusters = 0;

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (derived_should_stop(global_emptied_clusters)) { return; }

      auto &local_num_moved_nodes = num_moved_nodes.local();
      auto &local_rand = Randomize::instance();
      auto &local_rating_map = _rating_map.local();
      NodeID num_emptied_clusters = 0;

      const auto &chunk = _chunks[next_chunk.fetch_add(1, std::memory_order_relaxed)];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks = std::ceil(1.0 * (chunk.end - chunk.start) / kPermutationSize);
      std::vector<NodeID> sub_chunk_permutation(num_sub_chunks);
      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.end(), 0);
      local_rand.shuffle(sub_chunk_permutation);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < kPermutationSize; ++i) {
          const NodeID u = chunk.start + kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % kPermutationSize];
          if (u < chunk.end && _graph->degree(u) < _large_degree_threshold &&
              _active[u].load(std::memory_order_relaxed)) {
            _active[u].store(0, std::memory_order_relaxed);

            const auto [moved_node, emptied_cluster] = handle_node(u, local_rand, local_rating_map);
            if (moved_node) { ++local_num_moved_nodes; }
            if (emptied_cluster) { ++num_emptied_clusters; }
          }
        }
      }

      global_emptied_clusters += num_emptied_clusters;
    });

    const NodeID global_moved_nodes = num_moved_nodes.combine(std::plus{});
    return {global_moved_nodes, global_emptied_clusters};
  }

private:
  std::pair<bool, bool> handle_node(const NodeID u, Randomize &local_rand, RatingMap<EdgeWeight> &local_rating_map) {
    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived_cluster(u);
    const auto [new_cluster, new_gain] = find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    bool success = false;
    bool emptied_cluster = false;

    if (derived_cluster(u) != new_cluster) {
      // join new cluster without violating the maximum cluster size constraint
      if constexpr (Derived::kUseHardWeightConstraint) {
        ClusterWeight new_weight = _cluster_weights[new_cluster];

        // try to join this cluster but abort if it becomes full in the meantime
        while (new_weight + u_weight <= derived_max_cluster_weight(new_cluster)) {
          if (_cluster_weights[new_cluster].compare_exchange_weak(new_weight, new_weight + u_weight,
                                                                  std::memory_order_relaxed)) {
            success = true;
            break;
          }
        }
      } else {
        _cluster_weights[new_cluster].fetch_add(u_weight, std::memory_order_relaxed);
        success = true;
      }

      if (success) {
        _cluster_weights[u_cluster].fetch_sub(u_weight, std::memory_order_relaxed);
        if (Derived::kReportEmptyClusters && _cluster_weights[u_cluster].load(std::memory_order_relaxed) == 0) {
          emptied_cluster = true;
        }
        derived_set_cluster(u, new_cluster);
        activate_neighbors(u);
        IFSTATS(_expected_total_gain += new_gain);
      }
    }

    return {success, emptied_cluster};
  }

  std::pair<ClusterID, EdgeWeight> find_best_cluster(const NodeID u, const NodeWeight u_weight,
                                                     const ClusterID u_cluster, Randomize &local_rand,
                                                     RatingMap<EdgeWeight> &local_rating_map) {
    auto action = [&](auto &map) {
      const ClusterWeight initial_cluster_weight = _cluster_weights[u_cluster];
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
        const ClusterID v_cluster = derived_cluster(v);
        const EdgeWeight rating = _graph->edge_weight(e);
        map[v_cluster] += rating;
      };

      const EdgeID from = _graph->first_edge(u);
      const EdgeID to = from + std::min(_graph->degree(u), _max_num_neighbors);
      for (EdgeID e = from; e < to; ++e) { add_to_rating_map(e, _graph->edge_target(e)); }

      // the favored cluster is the one with the highest gain, independent of whether we can actually join that cluster
      ClusterID favored_cluster = u_cluster;

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating;
        state.current_cluster_weight = _cluster_weights[cluster].load(std::memory_order_relaxed);

        if constexpr (Derived::kUseFavoredCluster) {
          if (state.current_gain > state.best_gain) { favored_cluster = state.current_cluster; }
        }

        if (derived_accept_cluster(state)) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      // if we couldn't join any cluster, we keep the favored cluster in mind
      if constexpr (Derived::kUseFavoredCluster) {
        if (state.best_cluster == state.initial_cluster) { derived_set_favored_cluster(u, favored_cluster); }
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
    for (const NodeID v : _graph->adjacent_nodes(u)) { _active[v].store(1, std::memory_order_relaxed); }
  }

  // clang-format off
  void derived_reset_node_state(const NodeID u) { static_cast<Derived *>(this)->reset_node_state(u); }
  NodeID derived_cluster(const NodeID u) { return static_cast<Derived *>(this)->cluster(u); }
  void derived_set_cluster(const NodeID u, const ClusterID cluster) { static_cast<Derived *>(this)->set_cluster(u, cluster); }
  ClusterID derived_num_clusters() { return static_cast<Derived *>(this)->num_clusters(); }
  ClusterWeight derived_initial_cluster_weight(const ClusterID cluster) { return static_cast<Derived *>(this)->initial_cluster_weight(cluster); }
  ClusterWeight derived_max_cluster_weight(const ClusterID cluster) { return static_cast<Derived *>(this)->max_cluster_weight(cluster); }
  bool derived_accept_cluster(const ClusterSelectionState &state) { return static_cast<Derived *>(this)->accept_cluster(state); }
  // clang-format on

  void derived_set_favored_cluster(const NodeID u, const ClusterID favored_cluster) {
    if constexpr (Derived::kUseFavoredCluster) {
      static_cast<Derived *>(this)->set_favored_cluster(u, favored_cluster);
    }
  }

  [[nodiscard]] inline bool derived_should_stop(const NodeID num_emptied_clusters) {
    if constexpr (Derived::kControlProgress) {
      return static_cast<Derived *>(this)->should_stop(num_emptied_clusters);
    } else {
      return false;
    }
  }

protected:
  const NodeID _max_n;
  scalable_vector<parallel::IntegralAtomicWrapper<ClusterWeight>> _cluster_weights;
  Degree _large_degree_threshold{std::numeric_limits<Degree>::max()};
  Degree _max_num_neighbors{std::numeric_limits<Degree>::max()};

  const Graph *_graph{nullptr};

private:
  scalable_vector<parallel::IntegralAtomicWrapper<uint8_t>> _active;

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight>> _rating_map{[&] { return RatingMap<EdgeWeight>{_max_n}; }};
  RandomPermutations<NodeID, kPermutationSize, kNumberOfNodePermutations> _random_permutations{};

  parallel::IntegralAtomicWrapper<EdgeWeight> _expected_total_gain;
};
} // namespace kaminpar
