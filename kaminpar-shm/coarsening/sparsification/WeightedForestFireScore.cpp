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

namespace kaminpar::shm::sparsification {

WeightedForestFireScore::WeightedForestFireScore(
    const NetworKit::Graph &G, double pf, double targetBurntRatio
)
    : EdgeScore<double>(G),
      pf(pf),
      targetBurntRatio(targetBurntRatio) {}

void WeightedForestFireScore::run() {
  printf("hasEdgeIds: %d\n", G->hasEdgeIds());
  printf("not hasEdgeIds: %d\n", !G->hasEdgeIds());
  printf("hasEdgeIds equal false: %d\n", G->hasEdgeIds() == false);
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
      std::vector<std::pair<node, edgeid>> validEdges;
      G->forNeighborsOf(u, [&](node, node x, edgeid eid) {
        if (!visited[x]) {
          validEdges.emplace_back(x, eid);
        }
      });
      return validEdges;
    };

    count localEdgesBurnt = 0;

    while (!activeNodes.empty()) {
      node v = activeNodes.front();
      activeNodes.pop();

      std::vector<std::pair<node, edgeid>> validNeighbors = forwardNeighbors(v);
      std::vector<edgeweight> neighbor_weight_prefixsum(validNeighbors.size());
      edgeweight neighbour_weight_sum = 0;
      for (count i = 0; i < validNeighbors.size(); i++) {
        neighbour_weight_sum += G->getIthNeighborWeight(v, i);
        neighbor_weight_prefixsum[i] = neighbour_weight_sum;
      }

      while (true) {
        double q = Aux::Random::real(1.0);
        if (q > pf || validNeighbors.empty()) {
          break;
        }
        edgeweight r = Aux::Random::real(neighbour_weight_sum);
        count index = std::lower_bound(
                          neighbor_weight_prefixsum.begin(), neighbor_weight_prefixsum.end(), r
                      ) -
                      neighbor_weight_prefixsum.begin();

        { // mark node as visited, burn edge
          node x;
          edgeid eid;
          std::tie(x, eid) = validNeighbors[index];
          activeNodes.push(x);
#pragma omp atomic
          burnt[eid]++;
          localEdgesBurnt++;
          visited[x] = true;
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
