/*******************************************************************************
 * Generic label propagation benchmark for the shared-memory algorithm.
 *
 * @file:   shm_label_propagation_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   13.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  std::string graph_filename;
  int seed = 0;

  CLI::App app("Sparsified LP benchmark");
  app.add_option("graph", graph_filename, "Graph file")->required();
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-s,--seed", seed, "Seed for random number generation.")->default_val(seed);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  Random::reseed(seed);
  Graph graph = io::read(
      graph_filename, io::GraphFileFormat::METIS, ctx.node_ordering, ctx.compression.enabled
  );
  ctx.compression.setup(graph);
  ctx.partition.setup(graph, 2, 0.03);
  graph = graph::rearrange_by_degree_buckets(graph.csr_graph());
  if (ctx.edge_ordering == EdgeOrdering::COMPRESSION && !ctx.compression.enabled) {
    graph::reorder_edges_by_compression(graph.csr_graph());
  }

  StaticArray<NodeID> clustering(graph.n());
  timer::LocalTimer timer;

  NodeID num_deg0_nodes = 0;
  for (const NodeID u : graph.nodes()) {
    num_deg0_nodes += graph.degree(u) == 0;
  }
  LOG << "Number of deg0 nodes: " << num_deg0_nodes;

  LOG << "Graph,N,M,Threads,Threshold,NumVisited,NumSkipped,Time";
  const std::string graph_name = str::extract_basename(graph_filename);

  for (const double avg_degree :
       {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 1000.0}) {
    ctx.coarsening.clustering.lp.neighborhood_sampling_strategy =
        NeighborhoodSamplingStrategy::AVG_DEGREE;
    ctx.coarsening.clustering.lp.neighborhood_sampling_avg_degree_threshold = avg_degree;
    ctx.coarsening.clustering.lp.num_iterations = 1;
    ctx.coarsening.clustering.lp.impl = LabelPropagationImplementation::SINGLE_PHASE;

    LPClustering lp_clustering(ctx);
    lp_clustering.set_max_cluster_weight(compute_max_cluster_weight<NodeWeight>(
        ctx.coarsening, ctx.partition, graph.n(), graph.total_node_weight()
    ));
    lp_clustering.set_desired_cluster_count(0);

    timer.reset();
    lp_clustering.compute_clustering(clustering, graph, false);
    const double elapsed_s = timer.elapsed() / 1000.0 / 1000.0 / 1000.0;

    LOG << graph_name << "," << graph.n() << "," << graph.m() << "," << ctx.parallel.num_threads
        << "," << avg_degree << "," << lp_clustering.num_visited() << ","
        << lp_clustering.num_skipped() << "," << elapsed_s;
  }

  return 0;
}
