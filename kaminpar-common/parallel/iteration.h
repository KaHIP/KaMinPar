/*******************************************************************************
 * Reusable node iteration orders.
 *
 * @file:   iteration.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::iteration {

template <typename NodeID> struct NodeRange {
  NodeID from;
  NodeID to;
};

template <typename NodeID> struct Chunk {
  NodeID start;
  NodeID end;
};

struct Bucket {
  std::size_t start;
  std::size_t end;
};

template <typename NodeID> class NaturalNodeOrder {
public:
  explicit NaturalNodeOrder(const NodeRange<NodeID> range) : _range(range) {}

  [[nodiscard]] NodeRange<NodeID> range() const {
    return _range;
  }

  template <typename Visitor> void for_each(Visitor &&visitor) {
    for (NodeID u = _range.from; u < _range.to; ++u) {
      visitor(u);
    }
  }

  template <typename Visitor> void parallel_for_each(Visitor &&visitor) {
    tbb::parallel_for(tbb::blocked_range<NodeID>(_range.from, _range.to), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        visitor(u);
      }
    });
  }

private:
  NodeRange<NodeID> _range;
};

template <typename NodeID, std::size_t kPermutationSize, std::size_t kNumberOfNodePermutations>
class ChunkRandomNodeOrderWorkspace {
public:
  using NodePermutations = RandomPermutations<NodeID, kPermutationSize, kNumberOfNodePermutations>;

  NodePermutations &node_permutations() {
    return _node_permutations;
  }

  tbb::enumerable_thread_specific<std::vector<NodeID>> &sub_chunk_permutations() {
    return _sub_chunk_permutations;
  }

  tbb::enumerable_thread_specific<std::size_t> &num_chunks() {
    return _num_chunks;
  }

  tbb::enumerable_thread_specific<std::vector<Chunk<NodeID>>> &chunk_buffers() {
    return _chunk_buffers;
  }

  std::vector<Chunk<NodeID>> &chunks() {
    return _chunks;
  }

  std::vector<Bucket> &buckets() {
    return _buckets;
  }

  [[nodiscard]] bool
  has_order_for(const NodeRange<NodeID> range, const std::size_t min_chunk_size) const {
    return _has_order && _range.from == range.from && _range.to == range.to &&
           _min_chunk_size == min_chunk_size;
  }

  void mark_order(const NodeRange<NodeID> range, const std::size_t min_chunk_size) {
    _has_order = true;
    _range = range;
    _min_chunk_size = min_chunk_size;
  }

  void clear_order() {
    _chunks.clear();
    _buckets.clear();
    _has_order = false;
  }

  void free() {
    _chunks.clear();
    _chunks.shrink_to_fit();
    _buckets.clear();
    _buckets.shrink_to_fit();
    _sub_chunk_permutations.clear();
    _num_chunks.clear();
    _chunk_buffers.clear();
  }

private:
  NodePermutations _node_permutations;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _sub_chunk_permutations;
  tbb::enumerable_thread_specific<std::size_t> _num_chunks;
  tbb::enumerable_thread_specific<std::vector<Chunk<NodeID>>> _chunk_buffers;
  std::vector<Chunk<NodeID>> _chunks;
  std::vector<Bucket> _buckets;
  bool _has_order = false;
  NodeRange<NodeID> _range{};
  std::size_t _min_chunk_size = 0;
};

template <
    typename Graph,
    typename Workspace,
    typename NodeID = typename Graph::NodeID,
    typename EdgeID = typename Graph::EdgeID>
class ChunkRandomNodeOrder {
public:
  ChunkRandomNodeOrder(
      const Graph &graph,
      Workspace &workspace,
      const NodeRange<NodeID> range,
      const EdgeID min_chunk_size
  )
      : _graph(graph),
        _workspace(workspace),
        _range(range),
        _min_chunk_size(min_chunk_size) {}

  [[nodiscard]] NodeRange<NodeID> range() const {
    return _range;
  }

  void rebuild() {
    init_chunks();
  }

  template <typename Visitor> void for_each(Visitor &&visitor) {
    ensure_initialized();
    shuffle_chunks();

    auto &node_permutations = _workspace.node_permutations();
    for (const auto &chunk : _workspace.chunks()) {
      auto &rand = Random::instance();
      const auto &permutation = node_permutations.get(rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / permutation.size());
      std::vector<NodeID> sub_chunk_permutation(num_sub_chunks);
      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.end(), 0);
      rand.shuffle(sub_chunk_permutation);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < permutation.size(); ++i) {
          const NodeID u = chunk.start + permutation.size() * sub_chunk_permutation[sub_chunk] +
                           permutation[i % permutation.size()];
          if (u < chunk.end) {
            visitor(u);
          }
        }
      }
    }
  }

  template <typename Visitor> void parallel_for_each(Visitor &&visitor) {
    ensure_initialized();
    shuffle_chunks();

    parallel::Atomic<std::size_t> next_chunk = 0;
    auto &chunks = _workspace.chunks();
    auto &node_permutations = _workspace.node_permutations();
    auto &sub_chunk_permutations = _workspace.sub_chunk_permutations();

    tbb::parallel_for(static_cast<std::size_t>(0), chunks.size(), [&](const std::size_t) {
      auto &rand = Random::instance();
      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = chunks[chunk_id];
      const auto &permutation = node_permutations.get(rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / permutation.size());

      auto &sub_chunk_permutation = sub_chunk_permutations.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      rand.shuffle(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < permutation.size(); ++i) {
          const NodeID u = chunk.start + permutation.size() * sub_chunk_permutation[sub_chunk] +
                           permutation[i % permutation.size()];
          if (u < chunk.end) {
            visitor(u);
          }
        }
      }
    });
  }

private:
  void ensure_initialized() {
    if (!_workspace.has_order_for(_range, static_cast<std::size_t>(_min_chunk_size))) {
      init_chunks();
    }
  }

  void shuffle_chunks() {
    auto &chunks = _workspace.chunks();
    const auto &buckets = _workspace.buckets();
    tbb::parallel_for<std::size_t>(0, buckets.size(), [&](const std::size_t i) {
      const auto &bucket = buckets[i];
      Random::instance().shuffle(chunks.begin() + bucket.start, chunks.begin() + bucket.end);
    });
  }

  void init_chunks() {
    auto &chunks = _workspace.chunks();
    auto &buckets = _workspace.buckets();
    auto &chunk_buffers = _workspace.chunk_buffers();
    auto &num_chunks_ets = _workspace.num_chunks();

    chunks.clear();
    buckets.clear();
    _workspace.mark_order(_range, static_cast<std::size_t>(_min_chunk_size));

    const NodeID from = _range.from;
    const NodeID to = std::min(_range.to, _graph.n());

    const EdgeID max_chunk_size = std::max<EdgeID>(_min_chunk_size, std::sqrt(_graph.m()));
    const NodeID max_node_chunk_size = std::max<NodeID>(_min_chunk_size, std::sqrt(_graph.n()));

    NodeID position = 0;
    for (std::size_t bucket = 0; bucket < _graph.number_of_buckets(); ++bucket) {
      if (position + _graph.bucket_size(bucket) < from || _graph.bucket_size(bucket) == 0) {
        position += _graph.bucket_size(bucket);
        continue;
      }
      if (position >= to) {
        break;
      }

      NodeID remaining_bucket_size = _graph.bucket_size(bucket);
      if (from > _graph.first_node_in_bucket(bucket)) {
        remaining_bucket_size -= from - _graph.first_node_in_bucket(bucket);
      }
      const std::size_t bucket_size =
          std::min<NodeID>({remaining_bucket_size, to - position, to - from});

      parallel::Atomic<NodeID> offset = 0;
      const NodeID bucket_start = std::max(_graph.first_node_in_bucket(bucket), from);

      tbb::parallel_for(
          static_cast<int>(0), tbb::this_task_arena::max_concurrency(), [&](const int) {
            auto &local_chunks = chunk_buffers.local();
            auto &local_num_chunks = num_chunks_ets.local();

            while (offset < bucket_size) {
              const NodeID begin = offset.fetch_add(max_node_chunk_size);
              if (begin >= bucket_size) {
                break;
              }
              const NodeID end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

              EdgeID current_chunk_size = 0;
              NodeID chunk_start = bucket_start + begin;

              for (NodeID i = begin; i < end; ++i) {
                const NodeID u = bucket_start + i;
                current_chunk_size += _graph.degree(u);
                if (current_chunk_size >= max_chunk_size) {
                  local_chunks.push_back({chunk_start, u + 1});
                  chunk_start = u + 1;
                  current_chunk_size = 0;
                  ++local_num_chunks;
                }
              }

              if (chunk_start < bucket_start + end) {
                local_chunks.push_back({chunk_start, static_cast<NodeID>(bucket_start + end)});
                ++local_num_chunks;
              }
            }
          }
      );

      std::size_t num_chunks = 0;
      for (auto &local_num_chunks : num_chunks_ets) {
        num_chunks += local_num_chunks;
        local_num_chunks = 0;
      }

      const std::size_t chunks_start = chunks.size();
      parallel::Atomic<std::size_t> pos = chunks_start;
      chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(chunk_buffers.range(), [&](auto &r) {
        for (auto &local_chunks : r) {
          const std::size_t local_pos = pos.fetch_add(local_chunks.size());
          std::copy(local_chunks.begin(), local_chunks.end(), chunks.begin() + local_pos);
          local_chunks.clear();
        }
      });

      buckets.push_back({chunks_start, chunks.size()});
      position += _graph.bucket_size(bucket);
    }

    KASSERT(
        [&] {
          if (to < from) {
            return false;
          }
          std::vector<bool> hit(to - from);
          for (const auto &[start, end] : chunks) {
            KASSERT(start <= end, "");
            for (NodeID u = start; u < end; ++u) {
              KASSERT(from <= u, "");
              KASSERT(u < to, "");
              KASSERT(!hit[u - from], "");
              hit[u - from] = true;
            }
          }
          for (NodeID u = from; u < to; ++u) {
            KASSERT(hit[u - from], "");
          }
          return true;
        }(),
        "",
        assert::heavy
    );
  }

  const Graph &_graph;
  Workspace &_workspace;
  NodeRange<NodeID> _range;
  EdgeID _min_chunk_size;
};

} // namespace kaminpar::iteration
