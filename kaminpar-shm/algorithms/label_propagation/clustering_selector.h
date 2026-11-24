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
  [[nodiscard]] NodeID select_cluster(
      const NodeID from,
      const NodeWeight u_weight,
      Map &ratings,
      Random &random,
      TieContainer &ties
  ) const {
    constexpr std::size_t kMaxStackTies = 16;
    NodeID max_clusters[kMaxStackTies];
    EdgeWeight max_rating = 0;
    std::size_t max_count = 0;

    // Phase 1: L1 scan over ratings only (no random global weight loads).
    for (const auto [cluster, rating] : ratings.entries()) {
      if (rating > max_rating) {
        max_rating = rating;
        max_clusters[0] = cluster;
        max_count = 1;
      } else if (rating == max_rating) {
        if (max_count < kMaxStackTies) {
          max_clusters[max_count] = cluster;
        }
        ++max_count;
      }
    }

    if (max_count == 0) {
      return from;
    }

    const NodeWeight max_weight = _state.max_cluster_weight(from);
    const bool has_communities = _state.has_communities();
    auto is_feasible = [&](const NodeID cluster) {
      return (cluster == from || _state.cluster_weight(cluster) + u_weight <= max_weight) &&
             (!has_communities || _state.accepts_community_unchecked(from, cluster));
    };

    // Phase 2: Fast check on the highest-rated candidate(s) with exact rejection sampling.
    if (max_rating > 0 && max_count <= kMaxStackTies) {
      if (max_count == 1) {
        const NodeID cand = max_clusters[0];
        if (is_feasible(cand)) {
          return cand;
        }
      } else {
        std::size_t remaining = max_count;
        while (remaining > 0) {
          const std::size_t idx = (remaining == 1) ? 0 : random.random_index(0, remaining);
          const NodeID cand = max_clusters[idx];
          if (is_feasible(cand)) {
            return cand;
          }
          max_clusters[idx] = max_clusters[--remaining];
        }
      }
    }

    // Fallback: rare case when all top-rated candidates were rejected or >16 tied.
    NodeID best = from;
    EdgeWeight best_rating = 0;
    NodeID first_tie = from;
    std::size_t tie_count = 0;
    bool ties_used = false;

    for (const auto [cluster, rating] : ratings.entries()) {
      if (rating > best_rating) {
        if (is_feasible(cluster)) {
          best = cluster;
          best_rating = rating;
          first_tie = cluster;
          tie_count = 1;
        }
      } else if (rating == best_rating) {
        if (is_feasible(cluster)) {
          if (tie_count == 0) {
            first_tie = cluster;
            tie_count = 1;
          } else {
            if (tie_count == 1) {
              ties.clear();
              ties.push_back(first_tie);
              ties_used = true;
            }
            ties.push_back(cluster);
            ++tie_count;
          }
        }
      }
    }

    if (tie_count > 1) [[unlikely]] {
      best = ties[random.random_index(0, tie_count)];
    }
    if (ties_used) [[unlikely]] {
      ties.clear();
    }
    return best;
  }

  template <typename Map, typename TieContainer>
  [[nodiscard]] NodeID select_favored_cluster(
      const NodeID from, Map &ratings, Random &random, TieContainer &favored_ties
  ) const {
    constexpr std::size_t kMaxStackTies = 16;
    NodeID stack_ties[kMaxStackTies];
    NodeID favored = from;
    EdgeWeight favored_rating = 0;
    std::size_t tie_count = 0;
    bool ties_used = false;

    for (const auto [cluster, rating] : ratings.entries()) {
      if (rating > favored_rating) {
        favored_rating = rating;
        favored = cluster;
        stack_ties[0] = cluster;
        tie_count = 1;
        ties_used = false;
      } else if (rating == favored_rating) {
        if (tie_count < kMaxStackTies) {
          stack_ties[tie_count] = cluster;
          ++tie_count;
        } else {
          if (!ties_used) {
            favored_ties.clear();
            favored_ties.insert(favored_ties.end(), stack_ties, stack_ties + kMaxStackTies);
            ties_used = true;
          }
          favored_ties.push_back(cluster);
          ++tie_count;
        }
      }
    }

    if (tie_count > 1) [[unlikely]] {
      if (ties_used) [[unlikely]] {
        favored = favored_ties[random.random_index(0, tie_count)];
        favored_ties.clear();
      } else {
        favored = stack_ties[random.random_index(0, tie_count)];
      }
    }
    return favored;
  }

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
