/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#pragma once

#include <cmath>
#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

namespace jet {

// Cache the unfiltered strongest destination. A changing temperature or the
// afterburner filter must not turn a rejected move into a permanently cached result.
struct Destination {
  EdgeWeight source_connectivity = 0;
  EdgeWeight target_connectivity = 0;
  BlockID source = kInvalidBlockID;
  BlockID target = kInvalidBlockID;

  [[nodiscard]] EdgeWeight gain() const {
    return target_connectivity - source_connectivity;
  }

  [[nodiscard]] bool accepts(const double temperature) const {
    return target_connectivity > 0 &&
           (gain() >= 0 || gain() > -std::floor(temperature * source_connectivity));
  }
};

template <typename GainCache> class DestinationCache {
public:
  DestinationCache(GainCache &cache, const NodeID n) : _cache(cache), _destinations(n) {}

  const Destination &get(const NodeID u, const BlockID from) {
    auto &destination = _destinations[u];
    if (destination.target != kInvalidBlockID && destination.source == from) {
      return destination;
    }

    destination = {.source = from, .target = from};
    constexpr bool iterates_source = [] {
      if constexpr (requires { GainCache::kIteratesSourceBlock; }) {
        return GainCache::kIteratesSourceBlock;
      } else {
        return false;
      }
    }();
    if constexpr (!iterates_source || GainCache::kIteratesExactGains) {
      destination.source_connectivity = _cache.conn(u, from);
    }
    _cache.gains(u, from, [&](const BlockID to, auto &&value) {
      const EdgeWeight connectivity =
          value() + (GainCache::kIteratesExactGains ? destination.source_connectivity : 0);
      if (to == from) {
        destination.source_connectivity = connectivity;
      } else if (
          connectivity > destination.target_connectivity ||
          (connectivity == destination.target_connectivity && to < destination.target)
      ) {
        destination.target_connectivity = connectivity;
        destination.target = to;
      }
    });
    return destination;
  }

  // Invalidations can overlap during a parallel move phase. Queries only run
  // after that phase has joined, so the remaining fields need no atomics.
  void invalidate(const NodeID u) {
    __atomic_store_n(&_destinations[u].target, kInvalidBlockID, __ATOMIC_RELAXED);
  }

  template <typename Graph> void invalidate_neighborhood(const Graph &graph, const NodeID u) {
    invalidate(u);
    graph.adjacent_nodes(u, [&](const NodeID v) { invalidate(v); });
  }

  template <typename Graph> void invalidate_all(const Graph &graph) {
    graph.pfor_nodes([&](const NodeID u) { invalidate(u); });
  }

private:
  GainCache &_cache;
  StaticArray<Destination> _destinations;
};

} // namespace jet

template <typename Graph> class JetRefinerImpl;

class JetRefiner : public Refiner {
public:
  JetRefiner(const ExecutionContext &ctx);
  ~JetRefiner() override;

  JetRefiner(const JetRefiner &) = delete;
  JetRefiner &operator=(const JetRefiner &) = delete;

  JetRefiner(JetRefiner &&) noexcept = default;
  JetRefiner &operator=(JetRefiner &&) = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const ExecutionContext &_ctx;
  std::unique_ptr<ReifiedGraphComponent<JetRefinerImpl>> _impl;
};

} // namespace kaminpar::shm
