/*******************************************************************************
 * Adaptive per-vertex connectivity tables for Jet refinement.
 *
 * Implements the small-table idea in Section 4.3 of Gilbert et al.,
 * "Jet: Multilevel Graph Partitioning on Graphics Processing Units".
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
#include <type_traits>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/spin_mutex.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm {

// The initial rows share one allocation, but are sized from the number of adjacent blocks,
// not min(degree(u), k). Only rows that outgrow their initial slice allocate separately.
//
// Unlike the GPU algorithm's two update passes and third reconstruction pass, move() applies
// both deltas while holding the affected row's lock. Growing a row rehashes its exact current
// contents; reconstructing it from the partition here would race with concurrent moves.
// Queries also take this lock by default, permitting concurrent queries and moves. Disabling
// query locking requires that every query phase be separated from all move operations by a
// barrier; sequential query-then-move use is also allowed. initialize() and free() always require
// all users to be quiescent. Writer locks remain enabled for either specialization.
template <typename GraphType, bool lock_queries = true> class JetGainCache {
  struct UnlockedQueryGuard {
    explicit UnlockedQueryGuard(tbb::spin_mutex &) {}
  };
  using QueryGuard =
      std::conditional_t<lock_queries, tbb::spin_mutex::scoped_lock, UnlockedQueryGuard>;

  struct Entry {
    BlockID block = kInvalidBlockID;
    EdgeWeight weight = 0;
  };

  struct Row {
    Entry *entries = nullptr;
    std::size_t capacity = 0;
    EdgeWeight weighted_degree = 0;
    bool owns_entries = false;
    mutable tbb::spin_mutex mutex;

    ~Row() {
      if (owns_entries) {
        delete[] entries;
      }
    }
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
    _rows = std::make_unique<Row[]>(_n);

    StaticArray<std::size_t> offsets(static_cast<std::size_t>(_n) + 1);
    tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> buffers([&] {
      return FastResetArray<EdgeWeight>(_k);
    });
    graph.pfor_nodes([&](const NodeID u) {
      auto &buffer = buffers.local();
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        buffer[p_graph.block(v)] += weight;
        _rows[u].weighted_degree += weight;
      });
      const std::size_t count = buffer.size();
      offsets[u + 1] = count == 0 ? 0 : std::min(max_capacity(u), count + 1 + count / 4);
      buffer.clear();
    });
    parallel::prefix_sum(offsets.begin(), offsets.end(), offsets.begin());
    _initial_entries.resize(offsets.back());

    graph.pfor_nodes([&](const NodeID u) {
      Row &row = _rows[u];
      row.capacity = offsets[u + 1] - offsets[u];
      row.entries = row.capacity == 0 ? nullptr : _initial_entries.data() + offsets[u];

      auto &buffer = buffers.local();
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        buffer[p_graph.block(v)] += weight;
      });
      for (const auto [block, weight] : buffer.entries()) {
        insert(row, static_cast<BlockID>(block), weight);
      }
      buffer.clear();
    });
  }

  void free() {
    _rows.reset();
    _initial_entries.free();
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

  // With query locking enabled, the callback must not reenter this cache for the same vertex.
  template <typename Lambda> void for_each_connection(const NodeID u, Lambda &&lambda) const {
    const Row &row = _rows[u];
    const QueryGuard lock(row.mutex);
    for (std::size_t i = 0; i < row.capacity; ++i) {
      const Entry &entry = row.entries[i];
      if (entry.block != kInvalidBlockID) {
        lambda(entry.block, entry.weight);
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
      Row &row = _rows[v];
      const tbb::spin_mutex::scoped_lock lock(row.mutex);
      const std::size_t source = find(row, from);
      KASSERT(source < row.capacity && row.entries[source].weight >= weight);
      row.entries[source].weight -= weight;
      if (row.entries[source].weight == 0) {
        erase(row, source);
      }
      if (!insert(row, to, weight)) {
        grow(v, row);
        [[maybe_unused]] const bool inserted = insert(row, to, weight);
        KASSERT(inserted);
      }
    });
  }

  // Diagnostics include abandoned initial slices when reporting allocated storage. No concurrent
  // moves may run while collecting aggregate statistics.
  [[nodiscard]] std::size_t allocated_entries() const {
    std::size_t count = _initial_entries.size();
    for (NodeID u = 0; u < _n; ++u) {
      if (_rows[u].owns_entries) {
        count += _rows[u].capacity;
      }
    }
    return count;
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

  [[nodiscard]] static std::size_t hash(const BlockID block, const std::size_t capacity) {
    return (static_cast<std::uint64_t>(block) * 0x9e3779b97f4a7c15ULL) % capacity;
  }

  [[nodiscard]] static std::size_t next(const std::size_t i, const std::size_t capacity) {
    return i + 1 == capacity ? 0 : i + 1;
  }

  [[nodiscard]] static std::size_t find(const Row &row, const BlockID block) {
    if (row.capacity == 0) {
      return 0;
    }
    std::size_t pos = hash(block, row.capacity);
    for (std::size_t probes = 0; probes < row.capacity; ++probes) {
      const BlockID key = row.entries[pos].block;
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

  [[nodiscard]] static EdgeWeight lookup(const Row &row, const BlockID block) {
    const std::size_t pos = find(row, block);
    return pos == row.capacity ? 0 : row.entries[pos].weight;
  }

  // Probes are bounded even when the row is completely full.
  static bool insert(Row &row, const BlockID block, const EdgeWeight weight) {
    if (row.capacity == 0) {
      return false;
    }
    std::size_t pos = hash(block, row.capacity);
    for (std::size_t probes = 0; probes < row.capacity; ++probes) {
      Entry &entry = row.entries[pos];
      if (entry.block == block) {
        entry.weight += weight;
        return true;
      }
      if (entry.block == kInvalidBlockID) {
        entry = {block, weight};
        return true;
      }
      pos = next(pos, row.capacity);
    }
    return false;
  }

  // Backward-shift deletion keeps probe chains intact without accumulating tombstones.
  static void erase(Row &row, std::size_t hole) {
    row.entries[hole] = {};
    std::size_t pos = next(hole, row.capacity);
    for (std::size_t probes = 1; probes < row.capacity; ++probes) {
      const Entry entry = row.entries[pos];
      if (entry.block == kInvalidBlockID) {
        break;
      }
      const std::size_t home = hash(entry.block, row.capacity);
      const auto distance = [&](const std::size_t index) {
        return index >= home ? index - home : row.capacity - home + index;
      };
      if (distance(hole) < distance(pos)) {
        row.entries[hole] = entry;
        row.entries[pos] = {};
        hole = pos;
      }
      pos = next(pos, row.capacity);
    }
  }

  void grow(const NodeID u, Row &row) {
    const std::size_t capacity =
        std::min(max_capacity(u), std::max<std::size_t>(2, 2 * row.capacity));
    KASSERT(capacity > row.capacity);
    auto entries = std::make_unique<Entry[]>(capacity);
    Row replacement;
    replacement.entries = entries.get();
    replacement.capacity = capacity;
    for (std::size_t i = 0; i < row.capacity; ++i) {
      const Entry &entry = row.entries[i];
      if (entry.block != kInvalidBlockID) {
        [[maybe_unused]] const bool inserted = insert(replacement, entry.block, entry.weight);
        KASSERT(inserted);
      }
    }
    if (row.owns_entries) {
      delete[] row.entries;
    }
    row.entries = entries.release();
    row.capacity = capacity;
    row.owns_entries = true;
  }

  const Graph *_graph = nullptr;
  NodeID _n = 0;
  BlockID _k = 0;
  std::unique_ptr<Row[]> _rows;
  StaticArray<Entry> _initial_entries;
};

// Jet queries only between blocking parallel move phases. Avoiding reader locks prevents
// otherwise read-only scans from contending on frequently queried neighbors.
template <typename Graph> using SynchronousJetGainCache = JetGainCache<Graph, false>;

} // namespace kaminpar::shm
