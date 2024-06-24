/*
 * WeightedWeightedForestFireScore.cpp
 *
 *  Created on: 26.08.2014
 *      Author: Gerd Lindner
 */

#include "WeightedForestFireScore.hpp"

#include <limits>
#include <queue>
#include <set>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/graph/GraphTools.hpp>

#include "FiniteRandomDistribution.h"

namespace kaminpar::shm::sparsification {

WeightedForestFireScore::WeightedForestFireScore(
    const NetworKit::Graph &G, double pf, double targetBurntRatio
)
    : EdgeScore<double>(G),
      pf(pf),
      targetBurntRatio(targetBurntRatio) {}

void WeightedForestFireScore::run() {
  if (G->hasEdgeIds() == false) {
    throw std::runtime_error("edges have not been indexed - call indexEdges first");
  }

  std::vector<count> burnt(G->upperEdgeIdBound(), 0);
  count edgesBurnt = 0;

#pragma omp parallel
  while (edgesBurnt < targetBurntRatio * G->numberOfEdges()) {
    // Start a new fire
    std::queue<node> activeNodes;
    std::vector<bool> visited(G->upperNodeIdBound(), false);
    activeNodes.push(GraphTools::randomNode(*G));

    auto forwardNeighbors = [&](node u) {
      std::vector<edgetriple> validEdges;
      for (count i = 0; i < G->degree(u); i++) {
        auto [v, e] = G->getIthNeighborWithId(u, i);
        auto weight = G->getIthNeighborWeight(u, i);
        validEdges.emplace_back(v, e, weight);
      }
      return validEdges;
    };

    count localEdgesBurnt = 0;

    while (!activeNodes.empty()) {
      node v = activeNodes.front();
      activeNodes.pop();

      auto validNeighbors = forwardNeighbors(v);


      edgeweight validNeighboursWeightSum = std::accumulate(
          validNeighbors.begin(),
          validNeighbors.end(),
          0,
          [](edgeweight accumulator, edgetriple edge) { return accumulator + std::get<2>(edge); }
      );

      while (true) {
        double q = Aux::Random::real(1.0);
        if (q > pf || validNeighbors.empty()) {
          break;
        }
        edgeweight r = Aux::Random::real(validNeighboursWeightSum);
        count index = 0;
        edgeweight prefix_sum = std::get<2>(validNeighbors[0]);
        while ((prefix_sum += std::get<2>(validNeighbors[index])) < r) {
          index++;
        }

        { // mark node as visited, burn edge
          auto [x, eid, weight] = validNeighbors[index];
          activeNodes.push(x);
#pragma omp atomic
          burnt[eid]++;
          localEdgesBurnt++;
          visited[x] = true;
          validNeighboursWeightSum -= weight;
        }

        validNeighbors[index] = validNeighbors.back();
        validNeighbors.pop_back();
      }
    }

#pragma omp atomic
    edgesBurnt += localEdgesBurnt;
  }

  std::vector<double> burntNormalized(G->upperEdgeIdBound(), 0.0);
  double maxv = (double)*Aux::Parallel::max_element(std::begin(burnt), std::end(burnt));

  if (maxv > 0) {
#pragma omp parallel for
    for (omp_index i = 0; i < static_cast<omp_index>(burnt.size()); ++i) {
      burntNormalized[i] = burnt[i] / maxv;
    }
  }

  scoreData = std::move(burntNormalized);
  hasRun = true;
}

double WeightedForestFireScore::score(node, node) {
  throw std::runtime_error("Not implemented: Use scores() instead.");
}

double WeightedForestFireScore::score(edgeid) {
  throw std::runtime_error("Not implemented: Use scores() instead.");
}

} // namespace kaminpar::shm::sparsification
