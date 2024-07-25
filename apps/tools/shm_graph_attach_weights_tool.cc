/*******************************************************************************
 * Tool for assigning random weights based on different distributions to graphs
 * for the shared-memory algorithm.
 *
 * @file:   shm_graph_attach_weights_tool.cc
 * @author: Daniel Salwasser
 * @date:   30.06.2024
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <random>
#include <utility>

#include <tbb/concurrent_hash_map.h>
#include <tbb/global_control.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/loops.h"

#include "apps/io/shm_io.h"
#include "apps/io/shm_metis_parser.h"
#include "apps/io/shm_parhip_parser.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

namespace {

enum class WeightDistribution {
  UNIFORM,
  ALTERNATING
};

[[nodiscard]] std::unordered_map<std::string, WeightDistribution> get_weight_distributions() {
  return {
      {"uniform", WeightDistribution::UNIFORM},
      {"alternating", WeightDistribution::ALTERNATING},
  };
}

struct EdgeHasher {
  using Edge = std::pair<NodeID, NodeID>;

  [[nodiscard]] std::size_t operator()(const Edge &edge) const noexcept {
    return edge.first ^ (edge.second << 1);
  }

  [[nodiscard]] std::size_t hash(const Edge &edge) const noexcept {
    return edge.first ^ (edge.second << 1);
  }

  [[nodiscard]] bool equal(const Edge &a, const Edge &b) const noexcept {
    return a == b;
  }
};

template <typename Lambda>
[[nodiscard]] StaticArray<EdgeWeight>
generate_edge_weights(const CSRGraph &graph, Lambda &&edge_weight_generator_factory) {
  StaticArray<EdgeWeight> edge_weights(graph.m(), static_array::noinit);

  using Edge = std::pair<NodeID, NodeID>;
  using ConcurrentHashMap = tbb::concurrent_hash_map<Edge, EdgeWeight, EdgeHasher>;
  ConcurrentHashMap edge_weights_map(graph.m() / 2);

  parallel::deterministic_for<NodeID>(
      0,
      graph.n(),
      [&](const NodeID from, const NodeID to, const int cpu) {
        edge_weight_generator_factory(cpu, [&](auto &&edge_weight_generator) {
          for (NodeID u = from; u < to; ++u) {
            graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
              if (u <= v) {
                const EdgeWeight w = edge_weight_generator(e, u, v);
                edge_weights[e] = w;

                typename ConcurrentHashMap::accessor entry;
                edge_weights_map.insert(entry, std::make_pair(u, v));
                entry->second = w;
              }
            });
          }
        });
      }
  );

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
        if (u > v) {
          typename ConcurrentHashMap::const_accessor entry;
          edge_weights_map.find(entry, std::make_pair(v, u));

          const EdgeWeight w = entry->second;
          edge_weights[e] = w;
        }
      });
    }
  });

  return edge_weights;
}

[[nodiscard]] StaticArray<EdgeWeight> generate_uniform_edge_weights(
    const CSRGraph &graph, const int seed, const EdgeWeight min, const EdgeWeight max
) {
  return generate_edge_weights(graph, [&](const int cpu, auto &&edge_weight_fetcher) {
    const int local_seed = seed + cpu;
    std::mt19937 gen(local_seed);
    std::uniform_int_distribution<EdgeWeight> dist(min, max);

    edge_weight_fetcher([&](const EdgeID, const NodeID, const NodeID) {
      const EdgeWeight weight = dist(gen);
      return weight;
    });
  });
}

[[nodiscard]] StaticArray<EdgeWeight> generate_alternating_edge_weights(
    const CSRGraph &graph,
    const int seed,
    const EdgeWeight min_small_weights,
    const EdgeWeight max_small_weights,
    const EdgeWeight min_large_weights,
    const EdgeWeight max_large_weights
) {
  return generate_edge_weights(graph, [&](const int cpu, auto &&edge_weight_fetcher) {
    const int local_seed = seed + cpu;
    std::mt19937 gen(local_seed);
    std::uniform_int_distribution<EdgeWeight> small_dist(min_small_weights, max_small_weights);
    std::uniform_int_distribution<EdgeWeight> large_dist(min_large_weights, max_large_weights);

    edge_weight_fetcher([&](const EdgeID e, const NodeID, const NodeID) {
      const bool is_small_weight = (e % 2) == 0;

      if (is_small_weight) {
        const EdgeWeight weight = small_dist(gen);
        return weight;
      } else {
        const EdgeWeight weight = large_dist(gen);
        return weight;
      }
    });
  });
}

}; // namespace

int main(int argc, char *argv[]) {
  CLI::App app("Shared-memory graph attach-weights tool");

  std::string graph_filename;
  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS/ParHIP format")->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file format of the input graph:
  - metis
  - parhip)")
      ->capture_default_str();

  std::string weighted_graph_filename;
  GraphFileFormat weighted_graph_file_format = io::GraphFileFormat::METIS;
  app.add_option("--out", weighted_graph_filename, "Ouput file for storing the weighted graph")
      ->required();
  app.add_option("--out-f,--out-graph-file-format", weighted_graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file format used for storing the weighted graph:
  - metis
  - parhip)");

  int seed = 1;
  int num_threads = 1;
  app.add_option("-s,--seed", seed, "Seed for random number generation.")->capture_default_str();
  app.add_option("-t,--threads", num_threads, "Number of threads")->capture_default_str();

  WeightDistribution distribution;
  app.add_option("-d,--distribution", distribution)
      ->transform(CLI::CheckedTransformer(get_weight_distributions()).description(""))
      ->description(R"(Distribution used for generating edge weights:
  - uniform
  - alternating)")
      ->required()
      ->capture_default_str();

  EdgeWeight uniform_min_weight = 1;
  EdgeWeight uniform_max_weight = 32768;
  auto *uniform_group = app.add_option_group("Uniform Distribution");
  uniform_group->add_option("--u-min", uniform_min_weight, "Minimum weight value.")
      ->capture_default_str();
  uniform_group->add_option("--u-max", uniform_max_weight, "Maximum weight value.")
      ->capture_default_str();

  EdgeWeight alt_min_small_weights = 1;
  EdgeWeight alt_max_small_weights = 128;
  EdgeWeight alt_min_large_weights = 32768;
  EdgeWeight alt_max_large_weights = 8388608;
  auto *alt_group = app.add_option_group("Uniform Distribution");
  alt_group
      ->add_option("--a-min-small", alt_min_small_weights, "Minimum weight value of small weights.")
      ->capture_default_str();
  alt_group
      ->add_option("--a-max-small", alt_max_small_weights, "Maximum weight value of small weights.")
      ->capture_default_str();
  alt_group
      ->add_option("--a-min-large", alt_min_large_weights, "Minimum weight value of large weights.")
      ->capture_default_str();
  alt_group
      ->add_option("--a-max-large", alt_max_large_weights, "Maximum weight value of large weights.")
      ->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  LOG << "Reading input graph...";
  Graph graph = io::read(graph_filename, graph_file_format, false, false, NodeOrdering::NATURAL);
  CSRGraph &csr_graph = graph.csr_graph();

  LOG << "Generating edge weights...";
  StaticArray<EdgeWeight> edge_weights = [&] {
    switch (distribution) {
    case WeightDistribution::UNIFORM:
      return generate_uniform_edge_weights(csr_graph, seed, uniform_min_weight, uniform_max_weight);
    case WeightDistribution::ALTERNATING:
      return generate_alternating_edge_weights(
          csr_graph,
          seed,
          alt_min_small_weights,
          alt_max_small_weights,
          alt_min_large_weights,
          alt_max_large_weights
      );
    default:
      __builtin_unreachable();
    }
  }();

  Graph weighted_graph(std::make_unique<CSRGraph>(
      csr_graph.take_raw_nodes(),
      csr_graph.take_raw_edges(),
      csr_graph.take_raw_node_weights(),
      std::move(edge_weights)
  ));

  LOG << "Writing weighted graph...";
  switch (weighted_graph_file_format) {
  case GraphFileFormat::METIS:
    io::metis::write(weighted_graph_filename, weighted_graph);
    break;
  case GraphFileFormat::PARHIP:
    io::parhip::write(weighted_graph_filename, weighted_graph.csr_graph());
    break;
  }

  LOG << "Finished!";
  return EXIT_SUCCESS;
}
