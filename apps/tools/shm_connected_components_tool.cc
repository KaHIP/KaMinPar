/*******************************************************************************
 * Tool for fetching stats about connected components of a graph and for
 * extracting the largest connected component of a graph for the shared-memory
 * algorithm.
 *
 * @file:   shm_connected_components_tool.cc
 * @author: Daniel Salwasser
 * @date:   02.08.2024
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <bit>
#include <stack>
#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/bitvector_rank.h"
#include "kaminpar-common/logger.h"

#include "apps/io/shm_io.h"
#include "apps/io/shm_metis_parser.h"
#include "apps/io/shm_parhip_parser.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

namespace {

class Histogram {
public:
  using size_type = NodeID;

  [[nodiscard]] static size_type bucket(const size_type size) {
    KASSERT(size > 0);
    return (size == 1) ? 0 : (std::bit_width(size) - 1);
  }

  [[nodiscard]] static std::pair<size_type, size_type> bucket_range(const size_type num_bucket) {
    return std::make_pair(
        static_cast<size_type>(1) << num_bucket, static_cast<size_type>(1) << (num_bucket + 1)
    );
  }

  Histogram(const size_type max_size) : _buckets(bucket(max_size) + 1), _largest_bucket(0) {}

  void add(const size_type size) {
    const size_type num_bucket = bucket(size);
    _buckets[num_bucket] += 1;
    _largest_bucket = std::max(_largest_bucket, num_bucket);
  }

  [[nodiscard]] const std::vector<size_type> &buckets() const {
    return _buckets;
  }

  [[nodiscard]] const size_type largest_buckets() const {
    return _largest_bucket;
  }

private:
  std::vector<size_type> _buckets;
  size_type _largest_bucket;
};

struct ConnectedComponentsStats {
  NodeID num_connected_component;
  Histogram size_histogram;

  NodeID largest_connected_component_initial_node;
  NodeID largest_connected_component_order;
  EdgeID largest_connected_component_size;

  ConnectedComponentsStats(const NodeID num_nodes)
      : num_connected_component(0),
        size_histogram(num_nodes),
        largest_connected_component_initial_node(0),
        largest_connected_component_order(0),
        largest_connected_component_size(0) {}
};

template <typename Graph>
[[nodiscard]] ConnectedComponentsStats connected_components_stats(const Graph &graph) {
  ConnectedComponentsStats stats(graph.n());

  std::vector<bool> visited_nodes(graph.n());
  std::stack<NodeID> current_component;
  for (const NodeID node : graph.nodes()) {
    if (visited_nodes[node]) {
      continue;
    }

    NodeID current_component_order = 0;
    NodeID current_component_size = 0;

    visited_nodes[node] = true;
    current_component.push(node);
    do {
      const NodeID component_node = current_component.top();
      current_component.pop();

      current_component_order += 1;
      current_component_size += graph.degree(component_node);

      graph.adjacent_nodes(component_node, [&](const NodeID adjacent_node) {
        if (visited_nodes[adjacent_node]) {
          return;
        }

        visited_nodes[adjacent_node] = true;
        current_component.push(adjacent_node);
      });
    } while (!current_component.empty());

    stats.num_connected_component += 1;
    stats.size_histogram.add(current_component_order);
    if (current_component_order > stats.largest_connected_component_order) {
      stats.largest_connected_component_initial_node = node;
      stats.largest_connected_component_order = current_component_order;
      stats.largest_connected_component_size = current_component_size;
    }
  }

  return stats;
}

template <typename Graph>
[[nodiscard]] CSRGraph extract_largest_connected_component(
    const NodeID initial_node, const NodeID order, const EdgeID size, const Graph &graph
) {
  RankCombinedBitVector<> rank_bitvector(graph.n(), false);

  std::stack<NodeID> next_nodes;
  next_nodes.push(initial_node);
  do {
    const NodeID node = next_nodes.top();
    next_nodes.pop();

    graph.adjacent_nodes(node, [&](const NodeID adjacent_node) {
      if (rank_bitvector.is_set(adjacent_node)) {
        return;
      }

      rank_bitvector.set(adjacent_node);
      next_nodes.push(adjacent_node);
    });
  } while (!next_nodes.empty());

  rank_bitvector.update();

  StaticArray<EdgeID> nodes(order + 1, static_array::noinit);
  StaticArray<NodeID> edges(size, static_array::noinit);

  StaticArray<NodeWeight> node_weights;
  if (graph.is_node_weighted()) {
    node_weights.resize(order, static_array::noinit);
  }

  StaticArray<EdgeWeight> edge_weights;
  if (graph.is_edge_weighted()) {
    edge_weights.resize(size, static_array::noinit);
  }

  EdgeID cur_edge = 0;
  for (const NodeID node : graph.nodes()) {
    if (!rank_bitvector.is_set(node)) {
      continue;
    }

    const NodeID dense_node = rank_bitvector.rank(node);
    nodes[dense_node] = cur_edge;
    if (graph.is_node_weighted()) [[unlikely]] {
      node_weights[dense_node] = graph.node_weight(node);
    }

    graph.adjacent_nodes(node, [&](const NodeID adjacent_node, const EdgeWeight weight) {
      if (!rank_bitvector.is_set(adjacent_node)) {
        return;
      }

      const NodeID dense_adjacent_node = rank_bitvector.rank(adjacent_node);
      edges[cur_edge] = dense_adjacent_node;
      if (graph.is_edge_weighted()) [[unlikely]] {
        edge_weights[cur_edge] = weight;
      }

      cur_edge += 1;
    });
  }

  nodes[order] = size;
  return CSRGraph(
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      graph.sorted()
  );
}

}; // namespace

int main(int argc, char *argv[]) {
  CLI::App app("Shared-memory connected components tool");

  std::string graph_filename;
  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  bool compress = false;
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS/ParHIP format")->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file format of the input graph:
  - metis
  - parhip)")
      ->capture_default_str();
  app.add_flag("-c,--compress", compress, "Whether to compress the input graph")
      ->capture_default_str();

  std::string out_graph_filename;
  GraphFileFormat out_graph_file_format = io::GraphFileFormat::METIS;
  app.add_option(
      "--out",
      out_graph_filename,
      "Ouput file for storing the largest connected component of the input graph"
  );
  app.add_option("--out-f,--out-graph-file-format", out_graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file format used for storing the largest connected component:
  - metis
  - parhip)");

  int num_threads = 1;
  app.add_option("-t,--threads", num_threads, "Number of threads to use")->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
  const auto graph = io::read(graph_filename, graph_file_format, NodeOrdering::NATURAL, compress);

  auto stats = graph.reified([](const auto &graph) { return connected_components_stats(graph); });
  LOG << "Number of connected components: " << stats.num_connected_component;
  LOG << "Largest connected component: " << stats.largest_connected_component_order;

  LOG << "Size histogram:";
  const auto &histogram = stats.size_histogram.buckets();
  const auto largest_bucket = stats.size_histogram.largest_buckets();

  LOG << " Isolated nodes: " << histogram[0];
  for (std::size_t i = 1; i <= largest_bucket; ++i) {
    const auto [min, max] = Histogram::bucket_range(i);
    const auto num_components = histogram[i];
    LOG << " " << min << "-" << max << " nodes: " << num_components;
  }

  if (!out_graph_filename.empty()) {
    LOG << "Extracting largest connected component...";
    auto largest_connected_component = graph.reified([&](const auto &graph) {
      return extract_largest_connected_component(
          stats.largest_connected_component_initial_node,
          stats.largest_connected_component_order,
          stats.largest_connected_component_size,
          graph
      );
    });

    LOG << "Writing largest connected component...";
    switch (out_graph_file_format) {
    case GraphFileFormat::METIS:
      io::metis::write(
          out_graph_filename,
          Graph(std::make_unique<CSRGraph>(std::move(largest_connected_component)))
      );
      break;
    case GraphFileFormat::PARHIP:
      io::parhip::write(out_graph_filename, largest_connected_component);
      break;
    }
  }

  return EXIT_SUCCESS;
}
