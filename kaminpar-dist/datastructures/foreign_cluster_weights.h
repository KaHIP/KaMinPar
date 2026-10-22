/*******************************************************************************
 * Stable foreign-cluster indices with separately stored atomic weights.
 *
 * @file:   foreign_cluster_weights.h
 ******************************************************************************/
#pragma once

#include <atomic>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-dist/datastructures/growt.h"

namespace kaminpar::dist {

// All construction/reset/move operations require joined callers. Between construction phases,
// keys and indices are immutable; only the real signed64 weight objects change atomically.
class ForeignClusterWeights {
  using IndexMap = growt::GlobalNodeIDMap<GlobalNodeID>;
  using Handles = tbb::enumerable_thread_specific<typename IndexMap::handle_type>;
  static constexpr GlobalNodeID kPendingIndex = std::numeric_limits<GlobalNodeID>::max();

public:
  ForeignClusterWeights() = default;
  ForeignClusterWeights(const ForeignClusterWeights &) = delete;
  ForeignClusterWeights &operator=(const ForeignClusterWeights &) = delete;

  ForeignClusterWeights(ForeignClusterWeights &&other) : ForeignClusterWeights() {
    *this = std::move(other);
  }

  ForeignClusterWeights &operator=(ForeignClusterWeights &&other) {
    if (this != &other) {
      _build_handles.clear();
      _lookup_handles.clear();
      other._build_handles.clear();
      other._lookup_handles.clear();
      _indices = std::move(other._indices);
      _weights = std::move(other._weights);
      _ready = std::exchange(other._ready, false);
    }
    return *this;
  }

  void reset(const std::size_t initial_entries) {
    _build_handles.clear();
    _lookup_handles.clear();
    _indices = IndexMap{0};
    _weights.assign(initial_entries, 0);
    _ready = false;
  }

  // Initial indices are unique ghost-node offsets; this can run in parallel after reset().
  void initialize_entry(
      const std::size_t index, const GlobalNodeID cluster, const GlobalNodeWeight weight
  ) {
    KASSERT(!_ready && index < _weights.size());
    _weights[index] = weight;
    auto &handle = _build_handles.local();
    [[maybe_unused]] const bool inserted = handle.insert(cluster + 1, index).second;
    KASSERT(inserted, "Foreign cluster initialized twice");
  }

  // Join every initialize_entry() call before publishing the immutable index domain.
  void finish_initialization() {
    KASSERT(!_ready);
    _build_handles.clear();
    _ready = true;
  }

  // Cold batch only: no weight query/update may overlap this call. Candidate IDs may repeat.
  void prepare_keys(const std::span<const GlobalNodeID> clusters) {
    KASSERT(_ready);
    if (clusters.empty()) {
      return;
    }

    _ready = false;
    const std::size_t previous_size = _weights.size();
    KASSERT(clusters.size() <= _weights.max_size() - previous_size);
    _weights.resize(previous_size + clusters.size(), 0);
    std::atomic<std::size_t> next_index{previous_size};

    tbb::parallel_for<std::size_t>(0, clusters.size(), [&](const std::size_t i) {
      auto &handle = _build_handles.local();
      const GlobalNodeID key = clusters[i] + 1;
      if (handle.insert(key, kPendingIndex).second) {
        const std::size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
        KASSERT(index < _weights.size());
        _weights[index] = 0;
        [[maybe_unused]] const auto [it, found] = handle.update(
            key,
            [](auto &value, const GlobalNodeID assigned_index) { return value = assigned_index; },
            static_cast<GlobalNodeID>(index)
        );
        KASSERT(found && it != handle.end(), "New foreign cluster index disappeared");
      }
    });

    // Destruction flushes pending insertion counts and any synchronous growth before readers.
    _build_handles.clear();
    _weights.resize(next_index.load(std::memory_order_relaxed));
    _ready = true;
  }

  [[nodiscard]] GlobalNodeWeight weight(const GlobalNodeID cluster) {
    return __atomic_load_n(&_weights[index(cluster)], __ATOMIC_RELAXED);
  }

  void add(const GlobalNodeID cluster, const GlobalNodeWeight delta) {
    __atomic_fetch_add(&_weights[index(cluster)], delta, __ATOMIC_RELAXED);
  }

  void subtract(const GlobalNodeID cluster, const GlobalNodeWeight delta) {
    __atomic_fetch_sub(&_weights[index(cluster)], delta, __ATOMIC_RELAXED);
  }

  [[nodiscard]] std::size_t size() const {
    KASSERT(_ready);
    return _weights.size();
  }

private:
  [[nodiscard]] std::size_t index(const GlobalNodeID cluster) {
    KASSERT(_ready);
    auto &handle = _lookup_handles.local();
    const auto result = handle.find_quiescent(cluster + 1);
    KASSERT(result.has_value() && *result < _weights.size(), "Unknown foreign cluster");
    return static_cast<std::size_t>(*result);
  }

  IndexMap _indices{0};
  std::vector<GlobalNodeWeight> _weights;
  bool _ready = false;
  Handles _build_handles{[this] { return _indices.get_handle(); }};
  Handles _lookup_handles{[this] { return _indices.get_handle(); }};
};

} // namespace kaminpar::dist
