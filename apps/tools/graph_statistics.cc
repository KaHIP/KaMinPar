/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "arguments_parser.h"
#include "datastructure/graph.h"
#include "definitions.h"
#include "io.h"
#include "tools/graph_tools.h"
#include "utility/console_io.h"
#include "utility/utility.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>

using namespace kaminpar;
using namespace kaminpar::tool;
namespace ranges = std::ranges;

struct Options {
  std::string filename;
  bool verbose;
  bool fast;
  bool compute_connected_components;
  bool enumerate_connected_components;
  bool compute_k_cores;
  bool enumerate_k_cores;
  bool content_only;
  bool header_only;
  bool deg10_based_degree_distribution;
};

struct Statistics {
  std::string name;
  Degree max_degree;
  Degree min_degree;
  double avg_degree;
  double q1_degree;
  double med_degree;
  double q3_degree;
  double ninetieth_degree;
  std::array<Degree, std::numeric_limits<Degree>::digits + 1> degree_buckets;
  std::array<NodeWeight, std::numeric_limits<Degree>::digits + 1> degree_bucket_weights;
  Degree max_bucket;
  double density;

  std::vector<EdgeWeight> k_cores{};
  std::vector<KCoreStatistics> k_core_stats{};
  NodeID degeneration;

  std::vector<NodeID> cc_n;
  std::vector<NodeID> cc_2m;
  NodeID num_connected_components{0};
};

Options parse_args(int argc, char *argv[]) {
  Options opts{};

  // clang-format off
  Arguments args;
  args.positional()
      .argument("graph", "Graph in METIS format", &opts.filename)
      ;
  args.group("Optional")
      .argument("fast", "If set, only compute statistics based on the number of nodes and edges of the graph.", &opts.fast, 'f')
      .argument("verbose", "Print more information in a human readable format.", &opts.verbose, 'v')
      ;
  args.group("Statistic options")
      .argument("compute-ccs", "Compute connected components and print aggregated statistics.", &opts.compute_connected_components)
      .argument("enumerate-ccs", "... and print the size of each connected component.", &opts.enumerate_connected_components)
      .argument("compute-k-cores", "Compute k-cores and print some aggregated statistics.", &opts.compute_k_cores)
      .argument("enumerate-k-cores", "... and print the size of each k-core.", &opts.enumerate_k_cores)
      .argument("compute-degree-distribution-base10", "Compute degree distribution with buckets 1, 2, ..., 10, 20, ..., 100, 200, ..., 1000, ...", &opts.deg10_based_degree_distribution)
      ;
  args.group("Output options")
      .argument("content-only", "Do not output column names.", &opts.content_only)
      .argument("header-only", "Only output column names without any statistics.", &opts.header_only)
      ;
  args.parse(argc, argv, false);
  // clang-format on

  return opts;
}

void print_csv_header(const Options &opts) {
  if (opts.fast) {
    LOG << "Graph,N,M,Density,Version";
    return;
  }
  LLOG << "Graph,N,M,Density,Version,MinDegree,MaxDegree,AvgDegree,Q1Degree,MedDegree,Q3Degree,90thPercentileDegree,"
          "NumIsolatedNodes,DegreeBuckets";
  if (opts.compute_connected_components) {
    LLOG << ",NumberOfConnectedComponents,IsConnected";
    if (opts.enumerate_connected_components) { LLOG << ",ConnectedComponentsN,ConnectedComponentsM"; }
  }
  if (opts.compute_k_cores) {
    LLOG << ",Degeneracy";
    if (opts.enumerate_k_cores) { LLOG << ",CoresK,CoresN,CoresM,CoresDensity"; }
  }
  LOG;
}

void print_fast_statistics(const Statistics &stats, const Options &opts) {
  NodeID n;
  EdgeID m;
  bool has_node_weights;
  bool has_edge_weights;
  io::metis::read_format(opts.filename, n, m, has_node_weights, has_edge_weights);

  double density = 2.0 * m / n / (n - 1);

  LOG << stats.name << "," << n << "," << m << "," << density << "," << GIT_COMMIT_HASH;
}

void compute_node_statistics(const Graph &graph, Statistics &stats, const Options &) {
  std::vector<Degree> degrees(graph.n());
  ranges::transform(graph.nodes(), degrees.begin(), [&graph](const NodeID u) { return graph.degree(u); });
  ranges::sort(degrees);

  stats.max_degree = degrees.back();
  stats.min_degree = degrees.front();
  stats.avg_degree = static_cast<double>(graph.m()) / graph.n();
  stats.q1_degree = math::percentile(degrees, 0.25);
  stats.med_degree = math::percentile(degrees, 0.5);
  stats.q3_degree = math::percentile(degrees, 0.75);
  stats.ninetieth_degree = math::percentile(degrees, 0.9);
  stats.density = 1.0 * graph.m() / graph.n() / (graph.n() - 1);

  for (const NodeID u : graph.nodes()) {
    const auto bucket = degree_bucket(graph.degree(u));
    stats.max_bucket = std::max(stats.max_bucket, bucket);
    ++stats.degree_buckets[bucket];
    stats.degree_bucket_weights[bucket] += graph.node_weight(u);
  }
}

void compute_k_cores(const Graph &graph, Statistics &stats, const Options &) {
  EdgeWeight max_weighted_degree{0};
  for (const NodeID u : graph.nodes()) {
    EdgeWeight weighted_degree = 0;
    for (const EdgeID e : graph.incident_edges(u)) { weighted_degree += graph.edge_weight(e); }
    max_weighted_degree = std::max(max_weighted_degree, weighted_degree);
  }

  EdgeWeight k = 1;

  do {
    stats.k_cores = compute_k_core(graph, k, std::move(stats.k_cores));
    stats.k_core_stats.push_back(compute_k_core_statistics(graph, stats.k_cores));

    k = std::numeric_limits<EdgeWeight>::max();
    for (const auto &degree : stats.k_cores) {
      if (degree > 0) { k = std::min(degree + 1, k); }
    }
  } while (stats.k_core_stats.back().n > 0);

  stats.k_core_stats.pop_back(); // remove dummy entry
  stats.degeneration = stats.k_core_stats.back().k;
}

void compute_connected_components(const Graph &graph, Statistics &stats, const Options &opts) {
  const auto components = find_connected_components(graph);
  stats.num_connected_components = *std::max_element(components.begin(), components.end()) + 1;

  if (opts.enumerate_connected_components) {
    stats.cc_n.resize(stats.num_connected_components);
    stats.cc_2m.resize(stats.num_connected_components);

    for (const NodeID u : graph.nodes()) {
      ++stats.cc_n[components[u]];
      stats.cc_2m[components[u]] += graph.degree(u);
    }
  }
}

void print_verbose_deg10_based_distribution(const Graph &graph, const Statistics &stats) {
  const std::size_t width = (stats.max_degree == 0) ? 0 : std::floor(std::log10(stats.max_degree)) + 1;
  std::vector<NodeID> distribution(10 * width + 1);

  for (const NodeID u : graph.nodes()) {
    const Degree deg = graph.degree(u);
    const std::size_t digits = (deg == 0) ? 0 : std::floor(std::log10(deg)) + 1;
    std::size_t base = 10;
    for (std::size_t i = 1; i < digits; ++i) { base *= 10; }
    base /= 10;
    ++distribution[base + deg / base - 1];
  }

  for (std::size_t i = distribution.size(); i > 0; --i) {
    if (distribution[i - 1] != 0) { break; }
    distribution.pop_back();
  }

  LOG << "Log 10 degree buckets:";
  std::size_t from{0};
  std::size_t to{0};
  std::size_t inc{1};
  for (std::size_t i = 0; i < distribution.size(); ++i) {
    LOG << " - " << from << ".." << to << ": " << distribution[i];
    from = to + 1;
    to += inc;
    if (to % (inc * 10) == 0) {
      inc *= 10;
      to = 2 * inc - 1;
    }
  }
}

void print_verbose(const Graph &graph, Statistics &stats, const Options &opts) {
  LOG << "Degree buckets:";
  for (Degree bucket = 0; bucket <= stats.max_bucket; ++bucket) {
    LOG << " - bucket " << bucket << " with degrees " << lowest_degree_in_bucket(bucket) << ".."
        << lowest_degree_in_bucket(bucket + 1) - 1 << ": " << stats.degree_buckets[bucket]
        << " nodes = " << 100.0 * stats.degree_buckets[bucket] / graph.n() << " %,"
        << " avg_weight = " << 1.0 * stats.degree_bucket_weights[bucket] / stats.degree_buckets[bucket];
  }
  LOG;

  if (opts.enumerate_k_cores) {
    LOG << stats.degeneration << "-degenerate graph with k-core sizes:";
    for (const auto &core : stats.k_core_stats) {
      LOG << " - k=" << core.k << ": "
          << "n=" << core.n << " "
          << "m=" << core.m << " "
          << "density=" << (1.0 * core.m / core.n / (core.n - 1)) << " ";
    }
    LOG;
  }
}

void print_csv_content(const Graph &graph, const Statistics &stats, const Options &opts) {
  using namespace std::literals;
  logger::CompactContainerFormatter vec_formatter{";"sv};

  auto copy_transform_to_vec = [](const auto &container, auto &&transformer) {
    std::vector<decltype(transformer(container[0]))> vec(container.size());
    ranges::transform(container, vec.begin(), transformer);
    return vec;
  };

  LLOG << stats.name << ","                      //
       << graph.n() << ","                       //
       << graph.m() / 2 << ","                   //
       << stats.density << ","                   //
       << GIT_COMMIT_HASH << ","                 //
       << stats.min_degree << ","                //
       << stats.max_degree << ","                //
       << stats.avg_degree << ","                //
       << stats.q1_degree << ","                 //
       << stats.med_degree << ","                //
       << stats.q3_degree << ","                 //
       << stats.ninetieth_degree << ","          //
       << stats.degree_buckets[0] << ","         //
       << vec_formatter << stats.degree_buckets; //

  if (opts.compute_connected_components) {
    LLOG << ","                                    //
         << stats.num_connected_components << ","  //
         << (stats.num_connected_components == 1); //

    if (opts.enumerate_connected_components) {
      const auto cc_2m = copy_transform_to_vec(stats.cc_2m, [&](const EdgeID m) { return m / 2; });
      LLOG << ","                                //
           << vec_formatter << stats.cc_n << "," //
           << vec_formatter << cc_2m;            //
    }
  }

  if (opts.compute_k_cores) {
    LLOG << "," << stats.degeneration;
    if (opts.enumerate_k_cores) {
      auto core_ks = copy_transform_to_vec(stats.k_core_stats, [](const auto &core) { return core.k; });
      auto core_ns = copy_transform_to_vec(stats.k_core_stats, [](const auto &core) { return core.n; });
      auto core_ms = copy_transform_to_vec(stats.k_core_stats, [](const auto &core) { return core.m; });
      auto core_densities = copy_transform_to_vec(stats.k_core_stats, [](const auto &core) {
        return 1.0 * core.m / core.n / (core.n - 1);
      });

      LLOG << ","                              //
           << vec_formatter << core_ks << ","  //
           << vec_formatter << core_ns << ","  //
           << vec_formatter << core_ms << ","  //
           << vec_formatter << core_densities; //
    }
  }
  LOG;
}

int main(int argc, char *argv[]) {
  const Options opts = parse_args(argc, argv);

  if (opts.header_only) {
    print_csv_header(opts);
    return 0;
  }

  if (!std::ifstream(opts.filename)) {}
  if (opts.filename.empty()) { FATAL_ERROR << "Graph filename must be provided."; }

  Statistics stats{};
  stats.name = opts.filename.substr(opts.filename.find_last_of('/') + 1);

  if (opts.fast) {
    print_fast_statistics(stats, opts);
    return 0;
  }

  const Graph graph = io::metis::read(opts.filename);
  if (opts.verbose) { LOG << "Computing general statistics ..."; }
  compute_node_statistics(graph, stats, opts);

  if (opts.compute_k_cores) {
    if (opts.verbose) { LOG << "Computing k cores ..."; }
    compute_k_cores(graph, stats, opts);
  }
  if (opts.compute_connected_components) {
    if (opts.verbose) { LOG << "Computing connected components ..."; }
    compute_connected_components(graph, stats, opts);
  }

  // verbose / human readable output
  if (opts.verbose) { print_verbose(graph, stats, opts); }
  if (opts.deg10_based_degree_distribution) { print_verbose_deg10_based_distribution(graph, stats); }

  // csv output
  if (!opts.content_only) { print_csv_header(opts); }
  ALWAYS_ASSERT(!opts.header_only);
  print_csv_content(graph, stats, opts);

  return 0;
}
