/*******************************************************************************
 * Node-level semantics for balanced label-propagation refinement.
 *
 * @file:   balanced_node_processor.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <type_traits>

#include "kaminpar-shm/algorithms/label_propagation/balanced_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

template <typename Graph> class BalancedNodeProcessor {
public:
  using RatingMaps = RatingMapPool<BlockID, EdgeWeight, adaptive_rating_map::SparseMap>;
  using RatingMap = typename RatingMaps::RatingMap;

  struct LocalWorker {
    RatingMap &ratings;
    ScalableVector<BlockID> &ties;
    Random &random;
    RoundStats &stats;
  };

  BalancedNodeProcessor(
      const Graph &graph,
      BalancedState &state,
      RatingMaps &rating_maps,
      RoundStatistics &statistics,
      const BlockID num_blocks
  )
      : _graph(graph),
        _state(state),
        _rating_maps(rating_maps),
        _statistics(statistics),
        _num_blocks(num_blocks),
        _selector(state) {}

  [[nodiscard]] bool should_stop() const {
    return false;
  }

  [[nodiscard]] bool should_visit(const NodeID u) const {
    return _state.is_active(u);
  }

  [[nodiscard]] bool should_stop(LocalWorker &, const NodeID) const {
    return false;
  }

  [[nodiscard]] LocalWorker make_local_worker(Random &random) {
    return {
        _rating_maps.local_ratings(),
        _rating_maps.local_ties(),
        random,
        _statistics.local(),
    };
  }

  void visit(const NodeID u, LocalWorker &local) {
    visit_impl<false>(u, local);
  }

  void visit_first_iter(const NodeID u, LocalWorker &local) {
    visit_impl<true>(u, local);
  }

  void finish_work_unit(LocalWorker &) {}

private:
  template <bool kFirstIteration>
  void visit_impl(const NodeID u, LocalWorker &local) {
    if constexpr (std::is_same_v<std::decay_t<Graph>, CSRGraph>) {
      if (!_state.has_communities()) [[likely]] {
        visit_csr_unrestricted<kFirstIteration>(u, local);
        return;
      }
    }

    const NodeWeight u_weight = _graph.node_weight(u);
    const BlockID from = _state.cluster(u);
    const std::size_t upper_bound = std::min<BlockID>(_graph.degree(u), _num_blocks);
    const BlockID to = local.ratings.execute(upper_bound, [&](auto &ratings) {
      NeighborhoodRatings::accumulate(
          _graph,
          u,
          ratings,
          [&](const NodeID v) { return _state.cluster(v); },
          [&](const NodeID v) { return _state.accepts_neighbor(u, v); }
      );
      if constexpr (!kFirstIteration) {
        _state.deactivate(u);
      }
      const BlockID selected = _selector.select(from, u_weight, ratings, local.random, local.ties);
      ratings.clear();
      return selected;
    });

    if (to != from && _state.commit(_graph, u, from, to, u_weight)) {
      ++local.stats.moved;
    }
  }

  template <bool kFirstIteration>
  void visit_csr_unrestricted(const NodeID u, LocalWorker &local) {
    const EdgeID *__restrict__ g_nodes = _graph.raw_nodes().data();
    const NodeID *__restrict__ g_edges = _graph.raw_edges().data();
    const BlockID *__restrict__ part = _state.raw_partition();
    const EdgeID e_begin = g_nodes[u];
    const EdgeID e_end = g_nodes[u + 1];
    const BlockID from = __atomic_load_n(&part[u], __ATOMIC_RELAXED);

    const bool edge_weighted = _graph.is_edge_weighted();
    const EdgeWeight *__restrict__ g_ew = edge_weighted ? _graph.raw_edge_weights().data() : nullptr;

    // Loop 1: fast prefix scan while neighbors remain in block `from`.
    // Interior nodes exit here without touching the rating map, selector, or block weights.
    EdgeID e = e_begin;
    EdgeWeight internal_rating = 0;
    if (edge_weighted) {
      while (e < e_end) {
        const BlockID b = __atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED);
        if (b != from) {
          break;
        }
        internal_rating += g_ew[e];
        ++e;
      }
    } else {
      while (e < e_end) {
        const BlockID b = __atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED);
        if (b != from) {
          break;
        }
        ++e;
      }
      internal_rating = static_cast<EdgeWeight>(e - e_begin);
    }

    if (e == e_end) {
      if constexpr (!kFirstIteration) {
        _state.deactivate(u);
      }
      return;
    }

    // Loop 2: boundary node — accumulate remaining neighbors without interior-check branching.
    const NodeWeight u_weight = _graph.node_weight(u);
    const std::size_t upper_bound =
        std::min<BlockID>(static_cast<NodeID>(e_end - e_begin), _num_blocks);
    const BlockID to = local.ratings.execute(upper_bound, [&](auto &ratings) {
      if constexpr (requires { ratings.fast_accumulator(); }) {
        auto acc = ratings.fast_accumulator();
        if (internal_rating > 0) {
          acc.add(from, internal_rating);
        }
        if (edge_weighted) {
          for (; e < e_end; ++e) {
            acc.add(__atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED), g_ew[e]);
          }
        } else {
          for (; e < e_end; ++e) {
            acc.add(__atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED), 1);
          }
        }
      } else {
        if (internal_rating > 0) {
          ratings[from] = internal_rating;
        }
        if (edge_weighted) {
          for (; e < e_end; ++e) {
            ratings[__atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED)] += g_ew[e];
          }
        } else {
          for (; e < e_end; ++e) {
            ratings[__atomic_load_n(&part[g_edges[e]], __ATOMIC_RELAXED)] += 1;
          }
        }
      }

      if constexpr (!kFirstIteration) {
        _state.deactivate(u);
      }
      const BlockID selected = _selector.select(from, u_weight, ratings, local.random, local.ties);
      ratings.clear();
      return selected;
    });

    if (to != from && _state.commit(_graph, u, from, to, u_weight)) {
      ++local.stats.moved;
    }
  }
  const Graph &_graph;
  BalancedState &_state;
  RatingMaps &_rating_maps;
  RoundStatistics &_statistics;
  BlockID _num_blocks;
  BalancedSelector _selector;
};

} // namespace kaminpar::shm::lp
