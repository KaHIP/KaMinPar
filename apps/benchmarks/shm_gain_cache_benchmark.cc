/*******************************************************************************
 * Benchmark for gain caches.
 *
 * @file:   shm_gain_cache_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   27.02.2024
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hybrid_gain_cache.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/gains/tracing_gain_cache.h"

#include "kaminpar-common/timer.h"

#include "apps/benchmarks/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Create context with no preselected refinement algorithms
  Context ctx = create_default_context();
  ctx.refinement.algorithms.clear();

  // Parse CLI arguments
  std::string graph_filename;
  std::string partition_filename;
  std::string trace_filename;
  GainCacheStrategy gain_cache_strategy;
  bool is_sorted = false;
  int num_threads = 1;

  CLI::App app("Shared-memory FM benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-P,--partition", partition_filename, "Partition file")->required();
  app.add_option("-T,--trace", trace_filename, "Operation trace")->required();
  app.add_option("-C,--gc", gain_cache_strategy)
      ->transform(CLI::CheckedTransformer(get_gain_cache_strategies()).description(""))
      ->required();
  app.add_option("-t,--threads", num_threads, "Number of threads");

  app.add_flag("-S,--sorted", is_sorted)->capture_default_str();
  create_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  // Load input graph
  {
    auto input = load_partitioned_graph(graph_filename, partition_filename, is_sorted);

    ctx.partition.k = input.p_graph->k();
    ctx.parallel.num_threads = num_threads;
    ctx.setup(*input.graph);

    std::cout << "Replaying operations ..." << std::endl;
    std::vector<std::uint64_t> ans;

    switch (gain_cache_strategy) {
    case GainCacheStrategy::SPARSE:
      ans = tracing_gain_cache::replay<SparseGainCache<true>, GenericDeltaPartitionedGraph<>>(
          trace_filename, ctx, *input.p_graph
      );
      break;

    case GainCacheStrategy::ON_THE_FLY:
      ans = tracing_gain_cache::replay<OnTheFlyGainCache<true>, GenericDeltaPartitionedGraph<>>(
          trace_filename, ctx, *input.p_graph
      );
      break;

    case GainCacheStrategy::HYBRID:
      ans = tracing_gain_cache::replay<HybridGainCache<true>, GenericDeltaPartitionedGraph<>>(
          trace_filename, ctx, *input.p_graph
      );
      break;

    case GainCacheStrategy::DENSE:
      ans = tracing_gain_cache::replay<DenseGainCache<true>, GenericDeltaPartitionedGraph<>>(
          trace_filename, ctx, *input.p_graph
      );
      break;

    default:
      LOG_ERROR << "bad gain cache: " << gain_cache_strategy;
      break;
    }

    std::ofstream out(trace_filename + ".ans", std::ios_base::trunc | std::ios_base::binary);
    out.write(reinterpret_cast<const char *>(ans.data()), ans.size() * sizeof(std::uint64_t));
  }

  STOP_TIMER(); // Stop root timer
  Timer::global().print_human_readable(std::cout);

  return MPI_Finalize();
}

