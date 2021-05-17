/*! @file
 * Functions for loading and saving graphs and partitions.
 */
#pragma once

#include "algorithm/graph_utils.h"
#include "datastructure/graph.h"
#include "definitions.h"

namespace kaminpar::io {
namespace metis {
struct GraphInfo {
  NodeWeight total_node_weight;
  EdgeWeight total_edge_weight;
  bool has_isolated_nodes;
};

Graph read(const std::string &filename, bool ignore_node_weights = false, bool ignore_edge_weights = false);
GraphInfo read(const std::string &filename, StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
               StaticArray<NodeWeight> &node_weights, StaticArray<EdgeWeight> &edge_weights);
void read_format(const std::string &filename, NodeID &n, EdgeID &m, bool &has_node_weights, bool &has_edge_weights);
void write(const std::string &filename, const Graph &graph, const std::string &comment = "");
} // namespace metis

namespace partition {
void write(const std::string &filename, const std::vector<BlockID> &partition);
void write(const std::string &filename, const PartitionedGraph &p_graph);
void write(const std::string &filename, const StaticArray<BlockID> &partition, const NodePermutation &permutation);
void write(const std::string &filename, const PartitionedGraph &p_graph, const NodePermutation &permutation);
std::vector<BlockID> read(const std::string &filename);
} // namespace partition
} // namespace kaminpar::io