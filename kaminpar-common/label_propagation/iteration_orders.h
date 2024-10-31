/*******************************************************************************
 * Vertex iteration orders for label propagation.
 *
 * @file:   iteration_orders.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <atomic>
#include <limits>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/random.h"

namespace kaminpar {

template <typename Graph> class InOrderNodeIterator {
  using NodeID = typename Graph::NodeID;
  using EdgeID = typename Graph::EdgeID;

public:
  InOrderNodeIterator(const Graph &graph) : _graph(graph) {}

  template <typename Lambda> void operator()(const NodeID from, const NodeID to, Lambda &&lambda) {
    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(_graph.n(), to)),
        [&](const auto &r) { lambda(r); }
    );
  }

  template <typename Lambda> void operator()(Lambda &&lambda) {
    operator()(0, std::numeric_limits<NodeID>::max(), std::forward<Lambda>(lambda));
  }

private:
  const Graph &_graph;
};

template <typename Graph> class ChunkRandomizedNodeIterator {
  constexpr static std::size_t kPermutationSize = 64;
  constexpr static std::size_t kNumberOfNodePermutations = 64;
  constexpr static std::size_t kMinChunkSize = 1024;

  using NodeID = typename Graph::NodeID;
  using EdgeID = typename Graph::EdgeID;

  struct Chunk {
    Chunk(const NodeID begin, const NodeID end) : begin(begin), end(end) {}
    NodeID begin;
    NodeID end;
  };

  struct Bucket {
    Bucket(const std::size_t begin, const std::size_t end) : begin(begin), end(end) {}
    std::size_t begin;
    std::size_t end;
  };

public:
  ChunkRandomizedNodeIterator(const Graph &graph) : _graph(graph) {}

  template <typename Lambda> void operator()(const NodeID from, const NodeID to, Lambda &&lambda) {
    if (from != 0 || to != std::numeric_limits<NodeID>::max()) {
      _chunks.clear();
    }
    if (_chunks.empty()) {
      init_chunks(from, to);
    }
    shuffle_chunks();

    std::atomic<std::size_t> next_chunk = 0;

    tbb::parallel_for<std::size_t>(0, _chunks.size(), [&](std::size_t) {
      auto &local_rand = Random::instance();

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / kPermutationSize);

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.capacity() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      local_rand.shuffle(
          sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks
      );

      lambda([&](auto &&inner_lambda) {
        for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
          for (std::size_t i = 0; i < kPermutationSize; ++i) {
            const NodeID u = chunk.start + kPermutationSize * sub_chunk_permutation[sub_chunk] +
                             permutation[i % kPermutationSize];
            if (u >= chunk.end) {
              continue;
            }

            inner_lambda(u);
          }
        }
      });
    });
  }

  template <typename Lambda> void operator()(Lambda &&lambda) {
    operator()(0, std::numeric_limits<NodeID>::max(), std::forward<Lambda>(lambda));
  }

private:
  void init_chunks(const NodeID from, const NodeID to_limit) {
    const NodeID to = std::min(to_limit, _graph->n());

    _chunks.clear();
    _buckets.clear();

    const auto max_bucket =
        std::min<std::size_t>(math::floor_log2(_active_degree_limit), _graph->number_of_buckets());
    const auto max_chunk_size = std::max<EdgeID>(kMinChunkSize, std::sqrt(_graph->m()));
    const auto max_node_chunk_size = std::max<NodeID>(kMinChunkSize, std::sqrt(_graph->n()));

    NodeID position = 0;

    for (std::size_t bucket = 0; bucket < max_bucket; ++bucket) {
      if (position + _graph->bucket_size(bucket) < from || _graph->bucket_size(bucket) == 0) {
        position += _graph->bucket_size(bucket);
        continue;
      }

      if (position >= to) {
        break;
      }

      NodeID remaining_bucket_size = _graph->bucket_size(bucket);
      if (from > _graph->first_node_in_bucket(bucket)) {
        remaining_bucket_size -= from - _graph->first_node_in_bucket(bucket);
      }

      const auto bucket_start = std::max<std::size_t>(_graph->first_node_in_bucket(bucket), from);
      const auto bucket_size = std::min<NodeID>({remaining_bucket_size, to - position, to - from});

      std::atomic<NodeID> offset = 0;

      tbb::parallel_for<int>(0, tbb::this_task_arena::max_concurrency(), [&](int) {
        auto &chunks = _chunks_ets.local();
        auto &num_chunks = _num_chunks_ets.local();

        while (offset < bucket_size) {
          const auto begin = offset.fetch_add(max_node_chunk_size);
          const auto end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

          if (begin >= bucket_size) {
            break;
          }

          EdgeID current_chunk_size = 0;
          NodeID chunk_start = bucket_start + begin;

          for (NodeID i = begin; i < end; ++i) {
            const NodeID u = bucket_start + i;
            current_chunk_size += _graph->degree(u);
            if (current_chunk_size >= max_chunk_size) {
              chunks.push_back({chunk_start, u + 1});
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

      std::size_t num_chunks = 0;
      for (auto &local_num_chunks : _num_chunks_ets) {
        num_chunks += local_num_chunks;
        local_num_chunks = 0;
      }

      const std::size_t chunks_start = _chunks.size();
      std::atomic<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(_chunks_ets.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
          chunk.clear();
        }
      });

      _buckets.emplace_back(chunks_start, _chunks.size());
      position += _graph->bucket_size(bucket);
    }

    // Make sure that we cover all nodes in [from, to)
    KASSERT(
        [&] {
          std::vector<bool> hit(to - from);

          for (const auto &[start, end] : _chunks) {
            KASSERT(start <= end);

            std::int64_t total_work = 0;

            for (NodeID u = start; u < end; ++u) {
              KASSERT(from <= u && u < to);
              KASSERT(!hit[u - from]);

              hit[u - from] = true;
              total_work += _graph->degree(u);
            }
          }

          for (NodeID u = 0; u < to - from; ++u) {
            KASSERT(_graph->degree(u) == 0u || hit[u]);
          }

          return true;
        }(),
        "",
        assert::heavy
    );
  }

  void shuffle_chunks() {
    tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Random::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  using Permutations = RandomPermutations<NodeID, kPermutationSize, kNumberOfNodePermutations>;

  const Graph &_graph;

  NodeID _active_degree_limit = std::numeric_limits<NodeID>::max();

  Permutations &_random_permutations;

  tbb::enumerable_thread_specific<std::vector<NodeID>> _sub_chunk_permutation_ets;
  tbb::enumerable_thread_specific<std::size_t> _num_chunks_ets;
  tbb::enumerable_thread_specific<std::vector<Chunk>> _chunks_ets;

  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;
};

} // namespace kaminpar
