#pragma once
#include <oneapi/tbb/concurrent_vector.h>

#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {
class WeightedForestFireScore : public ScoreFunction<EdgeID> {
public:
  WeightedForestFireScore(double pf, double targetBurnRatio)
      : _pf(pf),
        _targetBurnRatio(targetBurnRatio) {}
  StaticArray<EdgeID> scores(const CSRGraph &g) override;

  static void make_scores_symetric(const CSRGraph &g, StaticArray<EdgeID> &scores);
  struct EdgeWithEndpoints {
    EdgeID edge_id;
    NodeID smaller_endpoint;
    NodeID larger_endpoint;
  };
  struct EdgeWithEndpointHasher {
    std::size_t operator()(const EdgeWithEndpoints &e) const {
      return hash(e);
    }
  };
  struct EdgeWithEnpointComparator {
    bool operator()(const EdgeWithEndpoints &e1, const EdgeWithEndpoints &e2) const {
      return e1.smaller_endpoint == e2.smaller_endpoint && e1.larger_endpoint == e2.larger_endpoint;
    }
  };
  static std::pair<NodeID, NodeID> as_pair(EdgeWithEndpoints edge_with_endpoints) {
    return {edge_with_endpoints.smaller_endpoint, edge_with_endpoints.larger_endpoint};
  }

  static u_int64_t hash(EdgeWithEndpoints edge) {
    uint64_t k = edge.smaller_endpoint + (static_cast<uint64_t>(edge.larger_endpoint) << 32);

    // MurrmurHash3 (https://github.com/aappleby/smhasher)
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53L;
    k ^= k >> 33;

    return k;
  }

private:
  double _pf;
  double _targetBurnRatio;

  void print_fire_statistics(const CSRGraph &g, EdgeID edges_burnt, int number_of_fires);
};
} // namespace kaminpar::shm::sparsification
