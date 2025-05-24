/*******************************************************************************
 * @file:   compact_hashing_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   13.08.2024
 ******************************************************************************/
#pragma once

#include <limits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/delta_gain_caches.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/compact_hash_map.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/aligned_prefix_sum.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <
    typename GraphType,
    template <typename> typename DeltaGainCacheType,
    bool iterate_nonadjacent_blocks,
    bool iterate_exact_gains = false>
class CompactHashingGainCache {
  SET_DEBUG(false);

  // Abuse MSB bit in the _weighted_degrees[] array for locking
  constexpr static UnsignedEdgeWeight kWeightedDegreeLock =
      (static_cast<UnsignedEdgeWeight>(1) << (std::numeric_limits<UnsignedEdgeWeight>::digits - 1));
  constexpr static UnsignedEdgeWeight kWeightedDegreeMask = ~kWeightedDegreeLock;

public:
  using Graph = GraphType;

  using Self = CompactHashingGainCache<
      GraphType,
      DeltaGainCacheType,
      iterate_nonadjacent_blocks,
      iterate_exact_gains>;

  using DeltaGainCache = DeltaGainCacheType<Self>;

  // If set to true, gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  CompactHashingGainCache(
      const Context &ctx, const NodeID preallocate_n, [[maybe_unused]] BlockID preallocate_k
  )
      : _ctx(ctx),
        // Since we do not know the size of the gain cache in advance (depends on vertex degrees),
        // we cannot preallocate it
        _gain_cache(0, static_array::noinit),
        _weighted_degrees(preallocate_n, static_array::noinit) {}

  void initialize(const Graph &graph, const PartitionedGraph &p_graph) {
    _graph = &graph;
    _p_graph = &p_graph;

    _n = _graph->n();
    _k = _p_graph->k();
    _bits_for_key = math::ceil_log2(_k);

    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(_n);
    }
    recompute_weighted_degrees();

    if (_offsets.size() < _n + 1) {
      SCOPED_TIMER("Allocation");
      _offsets.resize(_n + 1);
    }

    START_TIMER("Compute gain cache offsets");
    const std::size_t total_nbytes =
        parallel::aligned_prefix_sum(_offsets.begin(), _offsets.begin() + _n, [&](const NodeID u) {
          const std::uint64_t deg = math::ceil2(_graph->degree(u));
          const std::uint64_t width = compute_entry_width(u, deg < _k);
          const std::uint64_t nbytes = (deg < _k) ? width * deg : width * _k;
          return std::make_pair(width, nbytes);
        });
    STOP_TIMER();

    if (_n > 0) {
      const NodeID u = _n - 1;
      const std::uint64_t deg = math::ceil2(_graph->degree(u));
      const std::uint64_t width = compute_entry_width(u, deg < _k);
      const std::uint64_t nbytes = (deg < _k) ? width * deg : width * _k;

      if (width > 0) {
        _offsets[u] += (width - (_offsets[u] % width)) % width;
        KASSERT(_offsets[u] % width == 0u);
      }

      _offsets[u + 1] = _offsets[u] + nbytes;
    }

    KASSERT([&] {
      for (NodeID u : _graph->nodes()) {
        const EdgeID deg = math::ceil2(_graph->degree(u));
        const unsigned alignment = compute_entry_width(u, deg < _k);
        const unsigned nbytes = (deg < _k) ? alignment * deg : alignment * _k;
        KASSERT((alignment == 0u) || (_offsets[u] % alignment == 0u), "", assert::always);
        KASSERT(
            (_offsets[u + 1] - _offsets[u]) >= nbytes,
            "bad entry for " << u << " (n: " << _n << ")",
            assert::always
        );
        KASSERT(
            _offsets[u] <= _offsets[u + 1],
            "bad entry for " << u << " (n: " << _n << ")",
            assert::always
        );
      }
      return true;
    }());

    const std::size_t gain_cache_size = math::div_ceil(total_nbytes, sizeof(std::uint64_t));

    if (_gain_cache.size() < gain_cache_size) {
      SCOPED_TIMER("Allocation");
      _gain_cache.resize(gain_cache_size);
      DBG << "Allocating gain cache: " << _gain_cache.size() * sizeof(std::uint64_t) << " bytes";
    }

    DBG << "Gain cache summary: have " << _n << " nodes, " << _k << " blocks";
    DBG << "  Reserve " << (sizeof(UnsignedEdgeWeight) * 8 - _bits_for_key) << " of "
        << sizeof(UnsignedEdgeWeight) * 8 << " bits for gain values";
    DBG << "  Reserve " << _bits_for_key << " of " << sizeof(UnsignedEdgeWeight) * 8
        << " bits for block IDs";

    recompute_gains();
  }

  void free() {
    _gain_cache.free();
    _weighted_degrees.free();
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID block_from, const BlockID block_to) const {
    return conn(node, block_to) - conn(node, block_from);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return use_hash_table(node) ? conn_hash_table(node, block) : conn_full_table(node, block);
  }

  // Forcing inlining here seems to be very important
  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    if (use_hash_table(node)) {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_hash_table(node, from) : 0;

      if constexpr (kIteratesNonadjacentBlocks) {
        auto &buffer = _sparse_buffer_ets.local();

        with_hash_table(node, [&](const auto &&ht) {
          ht.for_each([&](const BlockID to, const EdgeWeight conn_to) { buffer.set(to, conn_to); });
        });

        for (BlockID to = 0; to < _k; ++to) {
          if (from != to) {
            lambda(to, [&] { return buffer.get(to) - conn_from; });
          }
        }

        buffer.clear();
      } else {
        with_hash_table(node, [&](const auto &&ht) {
          ht.for_each([&](const BlockID to, const EdgeWeight conn_to) {
            if (to != from) {
              lambda(to, [&] { return conn_to - conn_from; });
            }
          });
        });
      }
    } else {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_full_table(node, from) : 0;

      for (BlockID to = 0; to < _k; ++to) {
        if (from == to) {
          continue;
        }

        if constexpr (kIteratesNonadjacentBlocks) {
          lambda(to, [&] { return conn_full_table(node, to) - conn_from; });
        } else {
          const EdgeWeight conn_to = conn_full_table(node, to);
          if (conn_to > 0) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        }
      }
    }
  }

  KAMINPAR_INLINE void move(const NodeID node, const BlockID block_from, const BlockID block_to) {
    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      if (use_hash_table(v)) {
        lock(v);
        with_hash_table(v, [&](auto &&hash_table) {
          hash_table.decrease_by(block_from, weight);
          hash_table.increase_by(block_to, weight);
        });
        unlock(v);
      } else {
        with_full_table(v, [&](auto *full_table) {
          __atomic_fetch_sub(full_table + block_from, weight, __ATOMIC_RELAXED);
          __atomic_fetch_add(full_table + block_to, weight, __ATOMIC_RELAXED);
        });
      }
    });
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  is_border_node(const NodeID node, const BlockID block_of_node) const {
    return weighted_degree(node) != conn(node, block_of_node);
  }

  void print_statistics() const {
    // Nothing to do
  }

private:
  //
  // Init (mixed)
  //

  void recompute_gains() {
    SCOPED_TIMER("Reset gain cache");

    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) noexcept {
      _gain_cache[i] = 0;
    });
    _sparse_buffer_ets.clear();

    _graph->pfor_nodes([&](const NodeID u) {
      if (use_hash_table(u)) {
        with_hash_table(u, [&](auto &&hash_table) {
          _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
            hash_table.increase_by(_p_graph->block(v), static_cast<UnsignedEdgeWeight>(weight));
          });
        });
      } else {
        with_full_table(u, [&](auto *full_table) {
          _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
            full_table[_p_graph->block(v)] += static_cast<UnsignedEdgeWeight>(weight);
          });
        });
      }
    });
  }

  void recompute_weighted_degrees() {
    _graph->pfor_nodes([&](const NodeID u) {
      _weighted_degrees[u] = 0;

      // @todo avoid v lookup
      _graph->adjacent_nodes(u, [&](NodeID, const EdgeWeight weight) {
        _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
      });
    });
  }

  // Computes the number of bytes (0, 1, 2, 4, 8) required to store the entries for the given node
  KAMINPAR_INLINE int compute_entry_width(const NodeID node, const bool with_key) const {
    const auto max_value = static_cast<std::uint64_t>(weighted_degree(node));
    if (max_value == 0) {
      return 0;
    }

    const int bits = math::floor_log2(max_value) + 1 + (with_key ? _bits_for_key : 0);
    const int bytes = (bits + 7) / 8;
    const int ans = math::ceil2<unsigned>(bytes);
    KASSERT(ans == 1 || ans == 2 || ans == 4 || ans == 8);
    return ans;
  }

  [[nodiscard]] KAMINPAR_INLINE bool use_hash_table(const NodeID node) const {
    return math::ceil2(_graph->degree(node)) < _k;
  }

  //
  // Locking (hash table)
  //

  void lock(const NodeID node) {
    UnsignedEdgeWeight state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
    do {
      while (state & kWeightedDegreeLock) {
        state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
      }
    } while (!__atomic_compare_exchange_n(
        &_weighted_degrees[node],
        &state,
        state | kWeightedDegreeLock,
        false,
        __ATOMIC_ACQUIRE,
        __ATOMIC_RELAXED
    ));
  }

  void unlock(const NodeID node) {
    __atomic_store_n(
        &_weighted_degrees[node], _weighted_degrees[node] & kWeightedDegreeMask, __ATOMIC_RELAXED
    );
  }

  //
  // Lookups (hash table)
  //

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_hash_table(const NodeID node, const BlockID block) const {
    return with_hash_table(node, [&](const auto &&hash_table) {
      return static_cast<EdgeWeight>(hash_table.get(block));
    });
  }

  template <typename Lambda>
  KAMINPAR_INLINE decltype(auto) with_hash_table_impl(const NodeID node, Lambda &&l) const {
    const int width = compute_entry_width(node, true);
    const std::size_t start = _offsets[node];
    const std::size_t size = width > 0 ? math::floor2((_offsets[node + 1] - start) / width) : 0;

    KASSERT(use_hash_table(node));
    KASSERT(width == 0 || (start % width) == 0);
    KASSERT(width == 0 || (_offsets[node + 1] - start) % width == 0);
    KASSERT(
        math::is_power_of_2(size),
        "not a power of 2: " << size << "; start: " << start << "; width: " << width
                             << "; end: " << _offsets[node + 1]
    );

    switch (width) {
    case 1:
      return l(reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start, size);
      break;

    case 2:
      return l(
          reinterpret_cast<const std::uint16_t *>(
              reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
          ),
          size
      );
      break;

    case 4:
      return l(
          reinterpret_cast<const std::uint32_t *>(
              reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
          ),
          size
      );
      break;

    case 8:
      return l(
          reinterpret_cast<const std::uint64_t *>(
              reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
          ),
          size
      );
      break;
    }

    // Default case: isolated nodes with degree 0
    KASSERT(width == 0);
    return std::invoke_result_t<Lambda, const std::uint64_t *, std::size_t>();
  }

  template <typename Lambda>
  KAMINPAR_INLINE decltype(auto) with_hash_table(const NodeID node, Lambda &&l) const {
    return with_hash_table_impl(
        node,
        [&]<typename Width>(const Width *storage, const std::size_t size) {
          return l(CompactHashMap<const Width, true>(storage, size, _bits_for_key));
        }
    );
  }

  template <typename Lambda>
  KAMINPAR_INLINE decltype(auto) with_hash_table(const NodeID node, Lambda &&l) {
    return static_cast<const Self *>(this)->with_hash_table_impl(
        node,
        [&]<typename Width>(const Width *storage, const std::size_t size) {
          return l(CompactHashMap<Width, true>(const_cast<Width *>(storage), size, _bits_for_key));
        }
    );
  }

  //
  // Lookups (full table)
  //

  template <typename Lambda>
  KAMINPAR_INLINE decltype(auto) with_full_table(const NodeID node, Lambda &&l) const {
    const int width = compute_entry_width(node, false);
    const std::size_t start = _offsets[node];

    KASSERT(width == 0 || (start % width) == 0);

    switch (width) {
    case 1:
      return l(reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start);
      break;

    case 2:
      return l(reinterpret_cast<const std::uint16_t *>(
          reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
      ));
      break;

    case 4:
      return l(reinterpret_cast<const std::uint32_t *>(
          reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
      ));
      break;

    case 8:
      return l(reinterpret_cast<const std::uint64_t *>(
          reinterpret_cast<const std::uint8_t *>(_gain_cache.data()) + start
      ));
      break;
    }

    // Default case: isolated nodes with degree 0
    KASSERT(width == 0);
    return std::invoke_result_t<Lambda, const std::uint64_t *>();
  }

  template <typename Lambda>
  KAMINPAR_INLINE decltype(auto) with_full_table(const NodeID node, Lambda &&l) {
    return static_cast<const Self *>(this)->with_full_table(
        node, [&]<typename Entry>(const Entry *table) { l(const_cast<Entry *>(table)); }
    );
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_full_table(const NodeID node, const BlockID block) const {
    return with_full_table(node, [&](const auto *full_table) {
      return static_cast<EdgeWeight>(__atomic_load_n(full_table + block, __ATOMIC_RELAXED));
    });
  }

  //
  // Lookups (mixed)
  //

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight weighted_degree(const NodeID node) const {
    KASSERT(node < _weighted_degrees.size());
    return static_cast<EdgeWeight>(_weighted_degrees[node] & kWeightedDegreeMask);
  }

  const Context &_ctx;

  const Graph *_graph = nullptr;
  const PartitionedGraph *_p_graph = nullptr;

  NodeID _n = kInvalidNodeID;
  BlockID _k = kInvalidBlockID;

  StaticArray<std::uint64_t> _offsets;

  // Number of bits reserved in hash table cells to store the key (i.e., block ID) of the entry
  int _bits_for_key = 0;

  StaticArray<std::uint64_t> _gain_cache;
  StaticArray<UnsignedEdgeWeight> _weighted_degrees;

  mutable tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> _sparse_buffer_ets{[&] {
    return FastResetArray<EdgeWeight>(_k);
  }};
};

template <typename Graph>
using NormalCompactHashingGainCache = CompactHashingGainCache<Graph, GenericDeltaGainCache, true>;

template <typename Graph>
using LargeKCompactHashingGainCache =
    CompactHashingGainCache<Graph, LargeKGenericDeltaGainCache, false>;

} // namespace kaminpar::shm
