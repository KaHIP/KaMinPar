/*******************************************************************************
 * Standalone chunk-random iteration strategy for label propagation.
 *
 * Visits nodes in a randomized order based on degree-bucket-aware chunking:
 * 1. Nodes are split into chunks of roughly equal work (sum of degrees).
 * 2. Chunks within each degree bucket are shuffled.
 * 3. Within each chunk, sub-chunks are visited in random order, and nodes
 *    within a sub-chunk are visited using a random permutation.
 *
 * This is a standalone, independently testable component — no CRTP.
 *
 * @file:   chunk_random_iteration.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/label_propagation/config.h"

#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::lp {

template <typename Config> class ChunkRandomIterator {
public:
  using NodeID = shm::NodeID;
  using Chunk = AbstractChunk<NodeID>;
  using Permutations =
      RandomPermutations<NodeID, Config::kPermutationSize, Config::kNumberOfNodePermutations>;

  //! Data structures owned by this iterator that can be saved/restored for memory reuse.
  using DataStructures = std::tuple<
      tbb::enumerable_thread_specific<std::vector<NodeID>>,
      tbb::enumerable_thread_specific<std::size_t>,
      tbb::enumerable_thread_specific<std::vector<Chunk>>,
      std::vector<Chunk>,
      std::vector<Bucket>>;

  ChunkRandomIterator(Permutations &permutations) : _random_permutations(permutations) {}

  void setup(DataStructures structs) {
    auto [sub_chunk_permutation_ets, num_chunks_ets, chunks_ets, chunks, buckets] =
        std::move(structs);
    _sub_chunk_permutation_ets = std::move(sub_chunk_permutation_ets);
    _num_chunks_ets = std::move(num_chunks_ets);
    _chunks_ets = std::move(chunks_ets);
    _chunks = std::move(chunks);
    _buckets = std::move(buckets);
  }

  DataStructures release() {
    return std::make_tuple(
        std::move(_sub_chunk_permutation_ets),
        std::move(_num_chunks_ets),
        std::move(_chunks_ets),
        std::move(_chunks),
        std::move(_buckets)
    );
  }

  void free() {
    _chunks.clear();
    _chunks.shrink_to_fit();
    _buckets.clear();
    _buckets.shrink_to_fit();
  }

  void clear() {
    _chunks.clear();
    _buckets.clear();
  }

  [[nodiscard]] bool empty() const {
    return _chunks.empty();
  }

  /*!
   * Prepare for one LP iteration: build chunks on the first call, then shuffle them.
   *
   * Chunks are initialized only once per graph (when empty()) and reused across
   * iterations; each call re-shuffles them for a fresh random order.
   *
   * @param graph The graph.
   * @param from  First node in the iteration range.
   * @param to    One past the last node in the iteration range.
   * @param max_degree Maximum degree of nodes to include.
   */
  template <typename Graph>
  void prepare(const Graph &graph, const NodeID from, const NodeID to, const NodeID max_degree) {
    if (empty()) {
      init_chunks(graph, from, to, max_degree);
    }
    shuffle_chunks();
  }

  /*!
   * Initialize the chunks based on the graph's degree buckets.
   *
   * @param graph The graph.
   * @param from First node in the iteration range.
   * @param to One past the last node in the iteration range.
   * @param max_degree Maximum degree of nodes to include.
   */
  template <typename Graph>
  void init_chunks(const Graph &graph, NodeID from, NodeID to, const NodeID max_degree) {
    _chunks.clear();
    _buckets.clear();

    to = std::min(to, graph.n());

    const auto max_bucket =
        std::min<std::size_t>(math::floor_log2(max_degree), graph.number_of_buckets());
    const auto max_chunk_size =
        std::max<typename Graph::EdgeID>(Config::kMinChunkSize, std::sqrt(graph.m()));
    const NodeID max_node_chunk_size =
        std::max<NodeID>(Config::kMinChunkSize, std::sqrt(graph.n()));

    NodeID position = 0;
    for (std::size_t bucket = 0; bucket < max_bucket; ++bucket) {
      if (position + graph.bucket_size(bucket) < from || graph.bucket_size(bucket) == 0) {
        position += graph.bucket_size(bucket);
        continue;
      }
      if (position >= to) {
        break;
      }

      NodeID remaining_bucket_size = graph.bucket_size(bucket);
      if (from > graph.first_node_in_bucket(bucket)) {
        remaining_bucket_size -= from - graph.first_node_in_bucket(bucket);
      }
      const std::size_t bucket_size =
          std::min<NodeID>({remaining_bucket_size, to - position, to - from});

      parallel::Atomic<NodeID> offset = 0;

      const std::size_t bucket_start = std::max(graph.first_node_in_bucket(bucket), from);

      tbb::parallel_for(
          static_cast<int>(0), tbb::this_task_arena::max_concurrency(), [&](const int) {
            auto &chunks = _chunks_ets.local();
            auto &num_chunks = _num_chunks_ets.local();

            while (offset < bucket_size) {
              const NodeID begin = offset.fetch_add(max_node_chunk_size);
              if (begin >= bucket_size) {
                break;
              }
              const NodeID end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

              typename Graph::EdgeID current_chunk_size = 0;
              NodeID chunk_start = bucket_start + begin;

              for (NodeID i = begin; i < end; ++i) {
                const NodeID u = bucket_start + i;
                current_chunk_size += graph.degree(u);
                if (current_chunk_size >= max_chunk_size) {
                  chunks.push_back({chunk_start, u + 1});
                  chunk_start = u + 1;
                  current_chunk_size = 0;
                  ++num_chunks;
                }
              }

              if (current_chunk_size > 0) {
                chunks.push_back(
                    {static_cast<NodeID>(chunk_start), static_cast<NodeID>(bucket_start + end)}
                );
                ++num_chunks;
              }
            }
          }
      );

      std::size_t num_chunks = 0;
      for (auto &local_num_chunks : _num_chunks_ets) {
        num_chunks += local_num_chunks;
        local_num_chunks = 0;
      }

      const std::size_t chunks_start = _chunks.size();
      parallel::Atomic<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(_chunks_ets.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
          chunk.clear();
        }
      });

      _buckets.push_back({chunks_start, _chunks.size()});

      position += graph.bucket_size(bucket);
    }
  }

  /*!
   * Shuffle the chunks within each degree bucket.
   */
  void shuffle_chunks() {
    tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Random::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  /*!
   * Iterate over all nodes in chunk-random order, calling `handler(u)` for each
   * node. The handler returns `std::pair<bool, bool>` where the first element
   * indicates whether the node was moved and the second whether an empty cluster
   * was created.
   *
   * `should_stop()` is called periodically for early termination.
   * `is_active(u)` determines whether a node should be processed.
   *
   * @return The total number of nodes that were moved.
   */
  template <typename NodeHandler, typename ShouldStopFn, typename IsActiveFn, typename Graph>
  NodeID iterate(
      const Graph &graph,
      const NodeID max_degree,
      NodeHandler &&handler,
      ShouldStopFn &&should_stop,
      IsActiveFn &&is_active,
      parallel::Atomic<typename Config::ClusterID> &current_num_clusters
  ) {
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
    parallel::Atomic<std::size_t> next_chunk = 0;

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](std::size_t) {
      if (should_stop()) {
        return;
      }

      auto &local_num_moved_nodes = num_moved_nodes_ets.local();
      auto &local_rand = Random::instance();
      NodeID num_removed_clusters = 0;

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      local_rand.shuffle(
          sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks
      );

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start +
                           Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u >= chunk.end) {
            continue;
          }

          if (!is_active(u)) {
            continue;
          }

          const NodeID degree = graph.degree(u);
          if (degree < max_degree) {
            const auto [moved_node, emptied_cluster] = handler(u);

            if (moved_node) {
              ++local_num_moved_nodes;
            }
            if (emptied_cluster) {
              ++num_removed_clusters;
            }
          }
        }
      }

      current_num_clusters -= num_removed_clusters;
    });

    return num_moved_nodes_ets.combine(std::plus{});
  }

  /*!
   * First-phase iteration for the two-phase LP implementation. Same as iterate()
   * but the handler may defer high-degree nodes by returning nullopt.
   *
   * @return Pair of (total processed, total moved).
   */
  template <typename NodeHandler, typename ShouldStopFn, typename IsActiveFn, typename Graph>
  std::pair<NodeID, NodeID> iterate_first_phase(
      const Graph &graph,
      const NodeID max_degree,
      NodeHandler &&handler,
      ShouldStopFn &&should_stop,
      IsActiveFn &&is_active,
      parallel::Atomic<typename Config::ClusterID> &current_num_clusters
  ) {
    tbb::enumerable_thread_specific<NodeID> num_processed_nodes_ets;
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
    parallel::Atomic<std::size_t> next_chunk = 0;

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (should_stop()) {
        return;
      }

      auto &local_num_processed_nodes = num_processed_nodes_ets.local();
      auto &local_num_moved_nodes = num_moved_nodes_ets.local();
      auto &local_rand = Random::instance();
      NodeID num_removed_clusters = 0;

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      local_rand.shuffle(
          sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks
      );

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start +
                           Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u >= chunk.end) {
            continue;
          }

          if (!is_active(u)) {
            continue;
          }

          const NodeID degree = graph.degree(u);
          if (degree < max_degree) {
            ++local_num_processed_nodes;

            const auto [moved_node, emptied_cluster] = handler(u);
            if (moved_node) {
              ++local_num_moved_nodes;
            }
            if (emptied_cluster) {
              ++num_removed_clusters;
            }
          }
        }
      }

      current_num_clusters -= num_removed_clusters;
    });

    return std::make_pair(
        num_processed_nodes_ets.combine(std::plus{}), num_moved_nodes_ets.combine(std::plus{})
    );
  }

  [[nodiscard]] const std::vector<Chunk> &chunks() const {
    return _chunks;
  }

private:
  Permutations &_random_permutations;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _sub_chunk_permutation_ets;
  tbb::enumerable_thread_specific<std::size_t> _num_chunks_ets;
  tbb::enumerable_thread_specific<std::vector<Chunk>> _chunks_ets;
  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;
};

} // namespace kaminpar::lp
