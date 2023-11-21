/*******************************************************************************
 * Benchmark for the efficiency of interval encoding as a graph compression technique.
 *
 * @file:   shm_compressed_graph_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include <vector>

#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/variable_length_codec.h"

#include "apps/io/shm_io.cc"

using namespace kaminpar;
using namespace kaminpar::shm;

static std::string to_megabytes(std::size_t bytes) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
  return stream.str();
}

template <typename VarLengthCodec, bool IntervalEncoding, std::size_t kIntervalLengthTreshold = 3>
static std::size_t compressed_graph_size(const Graph &graph) {
  using NodeID = ::kaminpar::shm::NodeID;
  using NodeWeight = ::kaminpar::shm::NodeWeight;
  using EdgeID = ::kaminpar::shm::EdgeID;
  using EdgeWeight = ::kaminpar::shm::EdgeWeight;

  std::size_t used_bytes = 0;
  used_bytes += (graph.n() + 1) * sizeof(EdgeID);
  used_bytes += graph.raw_node_weights().size() * sizeof(NodeWeight);
  used_bytes += graph.raw_edge_weights().size() * sizeof(EdgeWeight);

  std::vector<NodeID> buffer;
  EdgeID first_edge = 0;
  for (const NodeID node : graph.nodes()) {
    const NodeID degree = graph.degree(node);

    if constexpr (IntervalEncoding) {
      used_bytes += VarLengthCodec::length_marker(degree);
    } else {
      used_bytes += VarLengthCodec::length(degree);
    }

    used_bytes += VarLengthCodec::length(first_edge);

    if (degree == 0) {
      continue;
    }

    first_edge += degree;
    for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
      buffer.push_back(adjacent_node);
    }

    // Sort the adjacent nodes in ascending order.
    std::sort(buffer.begin(), buffer.end());

    // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
    // least kIntervalLengthTreshold. Instead of storing all nodes, only store a representation of
    // the left extreme i and the length j - i + 1. Left extremes are compressed using the
    // differences between each left extreme and the previous right extreme minus 2 (because there
    // must be at least one integer between the end of an interval and the beginning of the next
    // one), except the first left extreme which is stored directly. The lengths are decremented by
    // kIntervalLengthTreshold, the minimum length of an interval.
    if constexpr (IntervalEncoding) {
      if (buffer.size() > 1) {
        std::size_t interval_count = 0;

        NodeID previous_right_extreme = 2;
        std::size_t interval_len = 1;
        NodeID prev_adjacent_node = *buffer.begin();
        for (auto iter = buffer.begin() + 1; iter != buffer.end(); ++iter) {
          const NodeID adjacent_node = *iter;

          if (prev_adjacent_node + 1 == adjacent_node) {
            interval_len++;

            // The interval ends if there are no more nodes or the next node is not the increment of
            // the current node.
            if (iter + 1 == buffer.end() || adjacent_node + 1 != *(iter + 1)) {
              if (interval_len >= kIntervalLengthTreshold) {
                const NodeID left_extreme = adjacent_node + 1 - interval_len;
                const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
                const std::size_t interval_length_gap = interval_len - kIntervalLengthTreshold;

                ++interval_count;
                used_bytes += VarLengthCodec::length(left_extreme_gap);
                used_bytes += VarLengthCodec::length(interval_length_gap);

                previous_right_extreme = adjacent_node;
                iter = buffer.erase(iter - interval_len + 1, iter + 1);
                if (iter == buffer.end()) {
                  break;
                }
              }

              interval_len = 1;
            }
          }

          prev_adjacent_node = adjacent_node;
        }

        if (interval_count > 0) {
          used_bytes += VarLengthCodec::length(interval_count);
        }

        // If all incident edges have been compressed using intervals then gap encoding cannot be
        // applied. Thus, go to the next node.
        if (buffer.empty()) {
          continue;
        }
      }
    }

    // Store the remaining adjacent node using gap encoding. That is instead of storing the nodes
    // v_1, v_2, ..., v_{k - 1}, v_k directly, store the gaps v_1 - u, v_2 - v_1, ..., v_k - v_{k -
    // 1} between the nodes, where u is the source node. Note that all gaps except the first one
    // have to be positive as we sorted the nodes in ascending order. Thus, only for the first gap
    // the sign is additionally stored.
    const NodeID first_adjacent_node = *buffer.begin();
    const std::make_signed_t<NodeID> first_gap = first_adjacent_node - node;
    used_bytes += VarLengthCodec::length_signed(first_gap);

    NodeID prev_adjacent_node = first_adjacent_node;
    const auto iter_end = buffer.end();
    for (auto iter = buffer.begin() + 1; iter != iter_end; ++iter) {
      const NodeID adjacent_node = *iter;
      const NodeID gap = adjacent_node - prev_adjacent_node;

      used_bytes += VarLengthCodec::length(gap);
      prev_adjacent_node = adjacent_node;
    }

    buffer.clear();
  }

  return used_bytes;
}

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;

  CLI::App app("Interval encoding efficiency benchmark");
  app.add_option("-G, --graph", graph_filename, "Graph file")->required();

  CLI11_PARSE(app, argc, argv);

  // Read input graph
  LOG << "Reading the input graph...";
  StaticArray<EdgeID> xadj;
  StaticArray<NodeID> adjncy;
  StaticArray<NodeWeight> vwgt;
  StaticArray<EdgeWeight> adjwgt;

  shm::io::metis::read<false>(graph_filename, xadj, adjncy, vwgt, adjwgt);

  const NodeID n = static_cast<NodeID>(xadj.size() - 1);
  const EdgeID m = xadj[n];

  StaticArray<EdgeID> nodes(xadj.data(), n + 1);
  StaticArray<NodeID> edges(adjncy.data(), m);
  StaticArray<NodeWeight> node_weights =
      (vwgt.empty()) ? StaticArray<NodeWeight>(0) : StaticArray<NodeWeight>(vwgt.data(), n);
  StaticArray<EdgeWeight> edge_weights =
      (adjwgt.empty()) ? StaticArray<EdgeWeight>(0) : StaticArray<EdgeWeight>(adjwgt.data(), m);

  Graph graph(std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
  ));

  // Run Benchmark
  LOG << "Calculating the compressed graph size...";
  std::size_t compressed_size = compressed_graph_size<VarIntCodec, false>(graph);
  std::size_t interval_compressed_size = compressed_graph_size<VarIntCodec, true>(graph);

  // Print the result summary
  LOG;
  cio::print_delimiter("Result Summary");

  LOG << "Input graph has " << graph.n() << " vertices and " << graph.m()
      << " edges. Its density is " << ((graph.m()) / (float)(graph.n() * (graph.n() - 1))) << ".";
  LOG;

  std::size_t graph_size = graph.raw_nodes().size() * sizeof(Graph::EdgeID) +
                           graph.raw_edges().size() * sizeof(Graph::NodeID) +
                           graph.raw_node_weights().size() * sizeof(Graph::NodeWeight) +
                           graph.raw_node_weights().size() * sizeof(Graph::EdgeWeight);
  LOG << "The uncompressed graph uses " << to_megabytes(graph_size) << " mb (" << graph_size
      << " bytes).";

  LOG << "The compressed graph without interval encoding uses " << to_megabytes(compressed_size)
      << " mb (" << compressed_size << " bytes).";

  LOG << "The compressed graph with interval encoding uses "
      << to_megabytes(interval_compressed_size) << " mb (" << interval_compressed_size
      << " bytes).";
  LOG;

  float compression_factor = graph_size / (float)compressed_size;
  LOG << "Thats a compression ratio of " << compression_factor << " without interval encoding.";

  float interval_compression_factor = graph_size / (float)interval_compressed_size;
  LOG << "Thats a compression ratio of " << interval_compression_factor
      << " with interval encoding.";

  float improvement = compressed_size / (float)interval_compressed_size;
  LOG << "Thats an improvement factor of " << improvement << " from interval encoding";
}
