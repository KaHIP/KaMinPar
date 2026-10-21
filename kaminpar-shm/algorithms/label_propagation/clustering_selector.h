/*******************************************************************************
 * Uniform cluster selection for label-propagation clustering.
 *
 * @file:   clustering_selector.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/uniform_tie_set.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

class ClusteringSelector {
public:
  struct Selection {
    NodeID cluster;
    NodeID favored_cluster;
  };

  struct RatedSelection {
    NodeID cluster;
    EdgeWeight rating;
    NodeID favored_cluster;
    EdgeWeight favored_rating;
  };

  explicit ClusteringSelector(const ClusteringStateView state) : _state(state) {}

  template <typename Map, typename TieContainer>
  [[nodiscard]] Selection select(
      const NodeID from,
      const NodeWeight u_weight,
      const bool select_favored,
      Map &ratings,
      Random &random,
      TieContainer &ties,
      TieContainer &favored_ties
  ) const {
    return select_impl<false>(from, u_weight, select_favored, ratings, random, ties, favored_ties);
  }

  template <typename Map, typename TieContainer>
  [[nodiscard]] RatedSelection select_rated(
      const NodeID from,
      const NodeWeight u_weight,
      const bool select_favored,
      Map &ratings,
      Random &random,
      TieContainer &ties,
      TieContainer &favored_ties
  ) const {
    return select_impl<true>(from, u_weight, select_favored, ratings, random, ties, favored_ties);
  }

private:
  template <bool kIncludeRatings, typename Map, typename TieContainer>
  [[nodiscard]] auto select_impl(
      const NodeID from,
      const NodeWeight u_weight,
      const bool select_favored,
      Map &ratings,
      Random &random,
      TieContainer &ties,
      TieContainer &favored_ties
  ) const {
    NodeID best = from;
    EdgeWeight best_rating = 0;
    NodeID favored = from;
    EdgeWeight favored_rating = 0;
    UniformTieSet best_ties(ties);
    UniformTieSet best_favored_ties(favored_ties);

    for (const auto [cluster, rating] : ratings.entries()) {
      if (select_favored) {
        if (rating > favored_rating) {
          favored_rating = rating;
          favored = cluster;
          best_favored_ties.replace_with(cluster);
        } else if (rating == favored_rating) {
          best_favored_ties.add(cluster);
        }
      }

      if (rating > best_rating) {
        if (accepts(cluster, from, u_weight)) {
          best = cluster;
          best_rating = rating;
          best_ties.replace_with(cluster);
        }
      } else if (rating == best_rating) {
        if (accepts(cluster, from, u_weight)) {
          best_ties.add(cluster);
        }
      }
    }

    best = best_ties.select_or(best, random);
    favored = best_favored_ties.select_or(favored, random);
    best_ties.clear();
    best_favored_ties.clear();

    if constexpr (kIncludeRatings) {
      return RatedSelection{
          .cluster = best,
          .rating = best_rating,
          .favored_cluster = select_favored ? favored : from,
          .favored_rating = select_favored ? favored_rating : 0,
      };
    } else {
      return Selection{
          .cluster = best,
          .favored_cluster = select_favored ? favored : from,
      };
    }
  }

  [[nodiscard]] bool
  accepts(const NodeID cluster, const NodeID from, const NodeWeight u_weight) const {
    return (_state.cluster_weight(cluster) + u_weight <= _state.max_cluster_weight(cluster) ||
            cluster == from) &&
           _state.accepts_community(from, cluster);
  }

  ClusteringStateView _state;
};

} // namespace kaminpar::shm::lp
