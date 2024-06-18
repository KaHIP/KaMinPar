/*
 * WeightedForestFireScore.hpp
 *
 *  Created on: 26.08.2014
 *      Author: Gerd Lindner
 */

#pragma once

#include <networkit/edgescores/EdgeScore.hpp>

namespace kaminpar::shm::sparsification {
using namespace NetworKit;

/**
 * Based on the Forest Fire algorithm introduced by Leskovec et al.
 * The burn frequency of the edges is used as edge score.
 */
class WeightedForestFireScore final : public EdgeScore<double> {

public:
  WeightedForestFireScore(const NetworKit::Graph &graph, double pf, double targetBurntRatio);
  double score(edgeid eid) override;
  double score(node u, node v) override;
  void run() override;

private:
  double pf;
  double targetBurntRatio;
};

} // namespace kaminpar::shm::sparsification
/* namespace NetworKit */
