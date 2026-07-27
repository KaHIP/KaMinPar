/*******************************************************************************
 * Node-level semantics for balanced label-propagation refinement.
 *
 * @file:   balanced_node_processor.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>

#include "kaminpar-shm/algorithms/label_propagation/balanced_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/algorithms/label_propagation/round_statistics.h"

#include "kaminpar-common/datastructures/sparse_map.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

template <typename Graph> class BalancedNodeProcessor {
public:
  using RatingMaps = RatingMapPool<EdgeWeight, BlockID, rm_backyard::SparseMap>;
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
      _state.deactivate(u);
      const BlockID selected = _selector.select(from, u_weight, ratings, local.random, local.ties);
      ratings.clear();
      return selected;
    });

    if (_state.commit(_graph, u, from, to, u_weight).moved) {
      ++local.stats.moved;
    }
  }

  void finish_work_unit(LocalWorker &) {}

private:
  const Graph &_graph;
  BalancedState &_state;
  RatingMaps &_rating_maps;
  RoundStatistics &_statistics;
  BlockID _num_blocks;
  BalancedSelector _selector;
};

} // namespace kaminpar::shm::lp
