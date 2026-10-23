/*******************************************************************************
 * Adaptive per-vertex connectivity tables for Jet refinement.
 *
 * Implements the small-table and batch-update ideas in Section 4.3 of Gilbert
 * et al., "Jet: Multilevel Graph Partitioning on Graphics Processing Units".
 *
 * @file:   jet_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/spin_mutex.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm {

struct JetGainCacheMove {
  NodeID node;
  BlockID from;
  BlockID to;
};

// Reserve min(degree(u), k) entries per row in reusable flat arrays, but only scan a small
// logical table determined by the number of adjacent blocks. Full k-entry rows use direct
// indexing. Keys and values are separate so small-batch updates can claim keys and add weights
// atomically without locking the row.
//
// move() remains an immediate, row-locked operation: it supports concurrent writers and, with
// lock_queries enabled, concurrent queries. A callback must not query the same row recursively.
// initialize(), free(), and move_batch() require quiescent users. In particular, move_batch()
// cannot overlap even locked queries: it updates atomic values without taking the row locks.
// Disabling query locks requires a barrier between every query phase and all move operations.
template <typename GraphType, bool lock_queries = true> class JetGainCache {
  static constexpr BlockID kDeletedBlock = kInvalidBlockID - 1;

  struct UnlockedQueryGuard {
    explicit UnlockedQueryGuard(tbb::spin_mutex &) {}
  };
  using QueryGuard =
      std::conditional_t<lock_queries, tbb::spin_mutex::scoped_lock, UnlockedQueryGuard>;

  struct Row {
    std::size_t offset = 0;
    std::size_t capacity = 0;
    EdgeWeight weighted_degree = 0;
    mutable tbb::spin_mutex mutex;
  };

public:
  using Graph = GraphType;
  constexpr static bool kIteratesNonadjacentBlocks = false;
  constexpr static bool kIteratesExactGains = false;

  JetGainCache(const Context &, NodeID, BlockID) {}

  JetGainCache(const JetGainCache &) = delete;
  JetGainCache &operator=(const JetGainCache &) = delete;

  void initialize(const Graph &graph, const PartitionedGraph &p_graph) {
    _graph = &graph;
    _k = p_graph.k();
    _n = graph.n();
    KASSERT(_k <= kDeletedBlock);
    if (_rows_size < _n) {
      _rows = std::make_unique<Row[]>(_n);
      _rows_size = _n;
    }
    if (_offsets.size() < static_cast<std::size_t>(_n) + 1) {
      _offsets.resize(static_cast<std::size_t>(_n) + 1, static_array::noinit);
    }
    _offsets[0] = 0;
    graph.pfor_nodes([&](const NodeID u) { _offsets[u + 1] = max_capacity(u); });
    parallel::prefix_sum(_offsets.begin(), _offsets.begin() + _n + 1, _offsets.begin());
    const std::size_t reserved = _offsets[_n];
    if (_keys.size() < reserved) {
      _keys.resize(reserved, static_array::noinit);
      _weights.resize(reserved, static_array::noinit);
    }
    if (_rebuild_flags.size() < _n) {
      _rebuild_flags.resize(_n, static_array::noinit);
    }
    graph.pfor_nodes([&](const NodeID u) {
      _rows[u].offset = _offsets[u];
      _rebuild_flags[u] = 0;
      rebuild(u, p_graph);
    });
  }

  void free() {
    _rows.reset();
    _rows_size = 0;
    _offsets.free();
    _keys.free();
    _weights.free();
    _rebuild_flags.free();
    _buffers.clear();
    _affected.clear();
    _graph = nullptr;
    _n = 0;
    _k = 0;
  }

  [[nodiscard]] EdgeWeight conn(const NodeID u, const BlockID block) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    return lookup(row, block);
  }

  [[nodiscard]] EdgeWeight gain(const NodeID u, const BlockID from, const BlockID to) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    return lookup(row, to) - lookup(row, from);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID u, const BlockID from, const std::pair<BlockID, BlockID> &targets) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    const EdgeWeight source = lookup(row, from);
    return {lookup(row, targets.first) - source, lookup(row, targets.second) - source};
  }

  [[nodiscard]] bool is_border_node(const NodeID u, const BlockID from) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    return row.weighted_degree != lookup(row, from);
  }

  template <typename Lambda> void for_each_connection(const NodeID u, Lambda &&lambda) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    for (std::size_t i = 0; i < row.capacity; ++i) {
      const EdgeWeight weight = _weights[row.offset + i];
      if (weight > 0) {
        lambda(row.capacity == _k ? static_cast<BlockID>(i) : _keys[row.offset + i], weight);
      }
    }
  }

  // Matches NormalSparseGainCache: yield connectivity (not gain), omitting the source block.
  template <typename Lambda> void gains(const NodeID u, const BlockID from, Lambda &&lambda) const {
    for_each_connection(u, [&](const BlockID to, const EdgeWeight weight) {
      if (to != from) {
        lambda(to, [weight] { return weight; });
      }
    });
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    if (from == to) {
      return;
    }
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (weight == 0) {
        return;
      }
      Row &row = _rows[v];
      const tbb::spin_mutex::scoped_lock lock(row.mutex);
      const std::size_t source = find(row, from);
      KASSERT(source < row.capacity && _weights[row.offset + source] >= weight);
      _weights[row.offset + source] -= weight;
      if (row.capacity != _k && _weights[row.offset + source] == 0) {
        _keys[row.offset + source] = kDeletedBlock;
      }
      if (!insert(row, to, weight)) {
        grow(v, row);
        [[maybe_unused]] const bool inserted = insert(row, to, weight);
        KASSERT(inserted);
      }
    });
  }

  // All partition labels must already reflect these moves; each vertex may appear at most
  // once. The two blocking parallel passes freeze the hash modulus and separate deletion from
  // insertion. If a small row overflows, only that row is rebuilt after the insertion barrier.
  // This avoids changing the modulus while concurrent insertions are still using it.
  void move_batch(const std::span<const JetGainCacheMove> moves, const PartitionedGraph &p_graph) {
    if (moves.empty()) {
      return;
    }
    KASSERT(p_graph.n() == _n && p_graph.k() == _k);
    if (moves.size() > _n / 10) {
      tbb::parallel_for<std::size_t>(0, moves.size(), [&](const std::size_t i) {
        const auto [u, from, to] = moves[i];
        if (from != to) {
          _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) { mark_for_rebuild(v); });
        }
      });
    } else {
      tbb::parallel_for<std::size_t>(0, moves.size(), [&](const std::size_t i) {
        const auto [u, from, to] = moves[i];
        if (from == to) {
          return;
        }
        _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
          if (weight == 0) {
            return;
          }
          const Row &row = _rows[v];
          const std::size_t source = atomic_find(row, from);
          KASSERT(source < row.capacity);
          const EdgeWeight previous =
              __atomic_fetch_sub(&_weights[row.offset + source], weight, __ATOMIC_RELAXED);
          KASSERT(previous >= weight);
          if (row.capacity != _k && previous == weight) {
            __atomic_store_n(&_keys[row.offset + source], kDeletedBlock, __ATOMIC_RELAXED);
          }
        });
      });
      tbb::parallel_for<std::size_t>(0, moves.size(), [&](const std::size_t i) {
        const auto [u, from, to] = moves[i];
        if (from == to) {
          return;
        }
        _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
          Row &row = _rows[v];
          // Once a small row overflows, its final-partition rebuild supersedes all remaining
          // updates. Avoid repeatedly probing that full row under heavy contention.
          if (weight == 0 ||
              (row.capacity != _k && __atomic_load_n(&_rebuild_flags[v], __ATOMIC_RELAXED) != 0)) {
            return;
          }
          if (!atomic_insert(row, to, weight)) {
            mark_for_rebuild(v);
          }
        });
      });
    }
    tbb::parallel_for_each(_affected, [&](auto &nodes) {
      tbb::parallel_for<std::size_t>(0, nodes.size(), [&](const std::size_t i) {
        const NodeID u = nodes[i];
        rebuild(u, p_graph);
        _rebuild_flags[u] = 0;
      });
      nodes.clear();
    });
  }

  // Backing storage is reused across initialize() calls, including calls with smaller graphs.
  // These aggregate diagnostics require quiescent users.
  [[nodiscard]] std::size_t allocated_entries() const {
    return _keys.size();
  }

  [[nodiscard]] std::size_t active_entries() const {
    std::size_t count = 0;
    for (NodeID u = 0; u < _n; ++u) {
      count += _rows[u].capacity;
    }
    return count;
  }

  [[nodiscard]] std::size_t row_capacity(const NodeID u) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    return row.capacity;
  }

private:
  [[nodiscard]] std::size_t max_capacity(const NodeID u) const {
    return std::min<std::size_t>(_graph->degree(u), _k);
  }

  [[nodiscard]] static std::size_t next(const std::size_t i, const std::size_t capacity) {
    return i + 1 == capacity ? 0 : i + 1;
  }

  template <bool atomic = false>
  [[nodiscard]] std::size_t find(const Row &row, const BlockID block) const {
    if (row.capacity == 0 || row.capacity == _k) {
      return row.capacity == 0 ? 0 : block;
    }
    std::size_t pos = block % row.capacity;
    for (std::size_t probes = 0; probes < row.capacity; ++probes) {
      const BlockID key = atomic ? __atomic_load_n(&_keys[row.offset + pos], __ATOMIC_RELAXED)
                                 : _keys[row.offset + pos];
      if (key == block) {
        return pos;
      }
      if (key == kInvalidBlockID) {
        break;
      }
      pos = next(pos, row.capacity);
    }
    return row.capacity;
  }

  [[nodiscard]] std::size_t atomic_find(const Row &row, const BlockID block) const {
    return find<true>(row, block);
  }

  [[nodiscard]] EdgeWeight lookup(const Row &row, const BlockID block) const {
    const std::size_t pos = find(row, block);
    return pos == row.capacity ? 0 : _weights[row.offset + pos];
  }

  // Search past tombstones before reusing one: the key may already occur later in its chain.
  // During a batch insertion phase slots only change from empty/deleted to occupied, so all
  // concurrent insertions of a new key choose the same first available slot or see that key.
  template <bool atomic = false>
  bool insert(Row &row, const BlockID block, const EdgeWeight weight) {
    std::size_t pos = find<atomic>(row, block);
    if (pos == row.capacity) {
      if (row.capacity == 0) {
        return false;
      }
      pos = block % row.capacity;
      for (std::size_t probes = 0; probes < row.capacity; ++probes) {
        BlockID &slot = _keys[row.offset + pos];
        BlockID key = atomic ? __atomic_load_n(&slot, __ATOMIC_RELAXED) : slot;
        if (key >= kDeletedBlock) {
          if constexpr (atomic) {
            __atomic_compare_exchange_n(
                &slot, &key, block, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
            );
          } else {
            slot = block;
          }
        }
        if (key == block || key >= kDeletedBlock) {
          break;
        }
        pos = next(pos, row.capacity);
        if (probes + 1 == row.capacity) {
          return false;
        }
      }
    }
    if constexpr (atomic) {
      __atomic_fetch_add(&_weights[row.offset + pos], weight, __ATOMIC_RELAXED);
    } else {
      _weights[row.offset + pos] += weight;
    }
    return true;
  }

  bool atomic_insert(Row &row, const BlockID block, const EdgeWeight weight) {
    return insert<true>(row, block, weight);
  }

  FastResetArray<EdgeWeight> &buffer() {
    auto &result = _buffers.local();
    if (result.capacity() < _k) {
      result.resize(_k);
    }
    return result;
  }

  void clear_row(const Row &row) {
    if (row.capacity == 0) {
      return;
    }
    std::fill_n(_keys.data() + row.offset, row.capacity, kInvalidBlockID);
    std::fill_n(_weights.data() + row.offset, row.capacity, EdgeWeight{0});
  }

  // One adjacency traversal builds the CPU thread's reusable dense scratch array. The final
  // compact table is then populated from its touched entries, not from a second graph scan.
  void rebuild(const NodeID u, const PartitionedGraph &p_graph) {
    Row &row = _rows[u];
    auto &scratch = buffer();
    row.weighted_degree = 0;
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (weight != 0) {
        scratch[p_graph.block(v)] += weight;
        row.weighted_degree += weight;
      }
    });
    const std::size_t count = scratch.size();
    row.capacity =
        count == 0 ? 0 : std::min(max_capacity(u), count + std::max<std::size_t>(3, count / 4));
    clear_row(row);
    for (const auto [block, weight] : scratch.entries()) {
      [[maybe_unused]] const bool inserted = insert(row, static_cast<BlockID>(block), weight);
      KASSERT(inserted);
    }
    scratch.clear();
  }

  void grow(const NodeID u, Row &row) {
    auto &scratch = buffer();
    for (std::size_t i = 0; i < row.capacity; ++i) {
      if (_weights[row.offset + i] > 0) {
        scratch[_keys[row.offset + i]] = _weights[row.offset + i];
      }
    }
    const std::size_t capacity =
        std::min(max_capacity(u), std::max<std::size_t>(4, 2 * row.capacity));
    KASSERT(capacity > row.capacity);
    row.capacity = capacity;
    clear_row(row);
    for (const auto [block, weight] : scratch.entries()) {
      [[maybe_unused]] const bool inserted = insert(row, static_cast<BlockID>(block), weight);
      KASSERT(inserted);
    }
    scratch.clear();
  }

  void mark_for_rebuild(const NodeID u) {
    if (__atomic_load_n(&_rebuild_flags[u], __ATOMIC_RELAXED) != 0) {
      return;
    }
    std::uint8_t expected = 0;
    if (__atomic_compare_exchange_n(
            &_rebuild_flags[u], &expected, 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )) {
      _affected.local().push_back(u);
    }
  }

  const Graph *_graph = nullptr;
  NodeID _n = 0;
  BlockID _k = 0;
  std::size_t _rows_size = 0;
  std::unique_ptr<Row[]> _rows;
  StaticArray<std::size_t> _offsets;
  StaticArray<BlockID> _keys;
  StaticArray<EdgeWeight> _weights;
  StaticArray<std::uint8_t> _rebuild_flags;
  tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> _buffers;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _affected;
};

// Jet queries only between blocking parallel move phases.
template <typename Graph> using SynchronousJetGainCache = JetGainCache<Graph, false>;

} // namespace kaminpar::shm
