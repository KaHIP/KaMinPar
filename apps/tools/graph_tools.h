#pragma once

#include "datastructure/graph.h"

#include <deque>
#include <functional>
#include <ranges>

namespace kaminpar::tool {
//
// k-core
//
struct KCoreStatistics {
  EdgeWeight k{std::numeric_limits<EdgeWeight>::max()};
  NodeID n{0};
  EdgeID m{0};
  NodeWeight max_node_weight{0};
  NodeWeight total_node_weight{0};
  EdgeWeight max_edge_weight{0};
  EdgeWeight total_edge_weight{0};
  Degree max_degree{0};
  EdgeWeight max_weighted_degree{0};
};

std::vector<NodeWeight> compute_k_core(const Graph &graph, EdgeWeight k, std::vector<EdgeWeight> core = {});

KCoreStatistics compute_k_core_statistics(const Graph &graph, const std::vector<EdgeWeight> &k_core);

std::vector<bool> k_core_to_indicator_array(const std::vector<EdgeWeight> &k_core);

//
// connected components
//
std::vector<BlockID> find_connected_components(const Graph &graph);
} // namespace kaminpar::tool