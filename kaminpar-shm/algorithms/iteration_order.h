/*******************************************************************************
 * Reusable parallel node iteration orders for shared-memory algorithms.
 *
 * @file:   iteration_order.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

namespace iteration_order_detail {

/*!
 * Kernel interface used by the iteration orders:
 *
 *   bool Kernel::should_stop() const;
 *   LocalKernel Kernel::make_local(Random &);
 *   bool LocalKernel::operator()(NodeID);
 *   bool LocalKernel::should_stop(NodeID);
 *   void LocalKernel::finish();
 *
 * The kernel object is shared between all tasks. A local kernel is created once
 * per TBB work unit and can therefore cache references to thread-local scratch
 * data without performing a TLS lookup for every node.
 */
template <typename Kernel>
using LocalKernel = decltype(std::declval<Kernel &>().make_local(std::declval<Random &>()));

} // namespace iteration_order_detail

/*!
 * Visits node IDs in ascending order within TBB blocked ranges.
 *
 * TBB can execute multiple ranges concurrently; hence, this component only
 * guarantees the order within each range, not one global sequential order.
 */
class InOrderIterationOrder {
public:
  void initialize(
      const NodeID num_nodes,
      const NodeID from = 0,
      const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    _from = std::min(from, num_nodes);
    _to = std::min(to, num_nodes);
    KASSERT(_from <= _to, "invalid node iteration range");
  }

  template <typename Kernel> void for_each(Kernel &kernel) const {
    tbb::parallel_for(tbb::blocked_range<NodeID>(_from, _to), [&](const auto &range) {
      if (kernel.should_stop()) {
        return;
      }

      Random &rand = Random::instance();
      iteration_order_detail::LocalKernel<Kernel> local = kernel.make_local(rand);

      for (NodeID u = range.begin(); u != range.end(); ++u) {
        if (local(u) && local.should_stop(u)) {
          break;
        }
      }

      local.finish();
    });
  }

private:
  NodeID _from = 0;
  NodeID _to = 0;
};

/*!
 * Visits nodes in degree-bucket-aware, chunk-shuffled order.
 *
 * The chunk plan is initialized once and reused across iterations. Before each
 * iteration, chunks are shuffled within their degree bucket. TBB tasks claim
 * chunks in shuffled order using one relaxed atomic counter. Within a chunk,
 * permutation-sized subchunks and the nodes in each subchunk are randomized.
 *
 * Nodes with degree zero are deliberately omitted from the chunk plan.
 */
class ChunkShuffledIterationOrder {
public:
  static constexpr NodeID kMinChunkSize = 1024;
  static constexpr NodeID kPermutationSize = 64;
  static constexpr std::size_t kNumberOfNodePermutations = 64;

  using Permutations = RandomPermutations<NodeID, kPermutationSize, kNumberOfNodePermutations>;

  explicit ChunkShuffledIterationOrder(Permutations &permutations)
      : _random_permutations(permutations) {}

  ChunkShuffledIterationOrder(const ChunkShuffledIterationOrder &) = delete;
  ChunkShuffledIterationOrder &operator=(const ChunkShuffledIterationOrder &) = delete;
  ChunkShuffledIterationOrder(ChunkShuffledIterationOrder &&) = delete;
  ChunkShuffledIterationOrder &operator=(ChunkShuffledIterationOrder &&) = delete;

  /*!
   * Builds and caches a chunk plan for the specified graph and node range.
   *
   * The maximum degree determines the last degree bucket included in the plan.
   * Fine-grained degree filtering remains the responsibility of the caller.
   */
  template <typename Graph>
  void initialize(
      const Graph &graph,
      const NodeID max_degree = std::numeric_limits<NodeID>::max(),
      const NodeID from = 0,
      const NodeID to = std::numeric_limits<NodeID>::max()
  ) {
    _chunks.clear();
    _buckets.clear();
    init_chunks(graph, max_degree, from, to);
  }

  void reset() {
    _chunks.clear();
    _buckets.clear();
  }

  void free() {
    _chunks.clear();
    _chunks.shrink_to_fit();
    _buckets.clear();
    _buckets.shrink_to_fit();

    _sub_chunk_permutation_ets.clear();
    _num_chunks_ets.clear();
    _chunks_ets.clear();
  }

  template <typename Kernel> void for_each(Kernel &kernel) {
    shuffle_chunks();

    parallel::Atomic<std::size_t> next_chunk = 0;
    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (kernel.should_stop()) {
        return;
      }

      Random &rand = Random::instance();
      iteration_order_detail::LocalKernel<Kernel> local = kernel.make_local(rand);

      const std::size_t chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const Chunk &chunk = _chunks[chunk_id];
      const auto &node_permutation = _random_permutations.get(rand);

      const std::size_t chunk_size = chunk.end - chunk.start;
      const std::size_t num_sub_chunks = (chunk_size + kPermutationSize - 1) / kPermutationSize;

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      rand.shuffle(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        const NodeID base = chunk.start + kPermutationSize * sub_chunk_permutation[sub_chunk];
        if (chunk.end - base >= kPermutationSize) {
          for (const NodeID offset : node_permutation) {
            local(base + offset);
          }
        } else {
          for (const NodeID offset : node_permutation) {
            const NodeID u = base + offset;
            if (u < chunk.end) {
              local(u);
            }
          }
        }
      }

      local.finish();
    });
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

  template <typename Graph>
  void init_chunks(
      const Graph &graph, const NodeID max_degree, const NodeID from, const NodeID requested_to
  ) {
    const NodeID to = std::min(requested_to, graph.n());
    KASSERT(from <= to, "invalid node iteration range");

    const std::size_t max_bucket =
        std::min<std::size_t>(math::floor_log2(max_degree), graph.number_of_buckets());
    const EdgeID max_chunk_size = std::max<EdgeID>(kMinChunkSize, std::sqrt(graph.m()));
    const NodeID max_node_chunk_size = std::max<NodeID>(kMinChunkSize, std::sqrt(graph.n()));

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
              const NodeID begin = offset.fetch_add(max_node_chunk_size, std::memory_order_relaxed);
              if (begin >= bucket_size) {
                break;
              }
              const NodeID end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

              EdgeID current_chunk_size = 0;
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
      tbb::parallel_for(_chunks_ets.range(), [&](auto &range) {
        for (auto &local_chunks : range) {
          const std::size_t local_pos =
              pos.fetch_add(local_chunks.size(), std::memory_order_relaxed);
          std::copy(local_chunks.begin(), local_chunks.end(), _chunks.begin() + local_pos);
          local_chunks.clear();
        }
      });

      _buckets.push_back({chunks_start, _chunks.size()});
      position += graph.bucket_size(bucket);
    }

    KASSERT(
        [&] {
          std::vector<bool> hit(to - from);
          for (const auto &[start, end] : _chunks) {
            KASSERT(start <= end);
            for (NodeID u = start; u < end; ++u) {
              KASSERT(from <= u);
              KASSERT(u < to);
              KASSERT(!hit[u - from]);
              hit[u - from] = true;
            }
          }

          for (NodeID i = 0; i < to - from; ++i) {
            KASSERT(graph.degree(from + i) == 0u || hit[i]);
          }
          return true;
        }(),
        "chunk plan does not cover every non-isolated node",
        assert::heavy
    );
  }

  void shuffle_chunks() {
    tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t i) {
      const Bucket &bucket = _buckets[i];
      Random::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  Permutations &_random_permutations;

  tbb::enumerable_thread_specific<std::vector<NodeID>> _sub_chunk_permutation_ets;
  tbb::enumerable_thread_specific<std::size_t> _num_chunks_ets;
  tbb::enumerable_thread_specific<std::vector<Chunk>> _chunks_ets;

  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;
};

} // namespace kaminpar::shm
