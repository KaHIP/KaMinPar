/*******************************************************************************
 * @file:   distributed_context.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "dkaminpar/context.h"

#include "dkaminpar/mpi_wrapper.h"

#include <tbb/parallel_for.h>

namespace dkaminpar {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {
    {PartitioningMode::KWAY, "kway"},
    {PartitioningMode::DEEP, "deep"},
    {PartitioningMode::RB, "rb"},
};

DEFINE_ENUM_STRING_CONVERSION(GlobalClusteringAlgorithm, global_clustering_algorithm) = {
    {GlobalClusteringAlgorithm::NOOP, "noop"},
    {GlobalClusteringAlgorithm::GLOBAL_LP, "global-lp"},
    {GlobalClusteringAlgorithm::REQUEST_LP, "request-lp"},
};

DEFINE_ENUM_STRING_CONVERSION(GlobalContractionAlgorithm, global_contraction_algorithm) = {
    {GlobalContractionAlgorithm::NO_MIGRATION, "no-migration"},
    {GlobalContractionAlgorithm::MINIMAL_MIGRATION, "minimal-migration"},
    {GlobalContractionAlgorithm::FULL_MIGRATION, "full-migration"},
};

DEFINE_ENUM_STRING_CONVERSION(InitialPartitioningAlgorithm, initial_partitioning_algorithm) = {
    {InitialPartitioningAlgorithm::KAMINPAR, "kaminpar"},
    {InitialPartitioningAlgorithm::RANDOM, "random"},
};

DEFINE_ENUM_STRING_CONVERSION(KWayRefinementAlgorithm, kway_refinement_algorithm) = {
    {KWayRefinementAlgorithm::NOOP, "noop"},
    {KWayRefinementAlgorithm::PROB_LP, "prob-lp"},
};

void LabelPropagationCoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "                                             //
      << prefix << "max_degree=" << large_degree_threshold << " "                                         //
      << prefix << "max_num_neighbors=" << max_num_neighbors << " "                                       //
      << prefix << "merge_singleton_clusters=" << merge_singleton_clusters << " "                         //
      << prefix << "merge_nonadjacent_clusters_threshold=" << merge_nonadjacent_clusters_threshold << " " //
      << prefix << "total_num_chunks=" << total_num_chunks << " "                                         //
      << prefix << "num_chunks=" << num_chunks << " "                                                     //
      << prefix << "min_num_chunks=" << min_num_chunks << " ";                                            //
}

void LabelPropagationRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "        //
      << prefix << "total_num_chunks=" << total_num_chunks << " "    //
      << prefix << "num_chunks=" << num_chunks << " "                //
      << prefix << "min_num_chunks=" << min_num_chunks << " "        //
      << prefix << "num_move_attempts=" << num_move_attempts << " "; //
}

void CoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "use_local_clustering" << use_local_clustering << " "                  //
      << prefix << "use_global_clustering=" << use_global_clustering << " "               //
      << prefix << "global_clustering_algorithm=" << global_clustering_algorithm << " "   //
      << prefix << "global_contraction_algorithm=" << global_contraction_algorithm << " " //
      << prefix << "contraction_limit=" << contraction_limit << " "                       //
      << prefix << "cluster_weight_limit=" << cluster_weight_limit << " "                 //
      << prefix << "cluster_weight_multiplier=" << cluster_weight_multiplier << " ";      //
  local_lp.print(out, prefix + "local_lp.");
  global_lp.print(out, prefix + "global_lp.");
}

void InitialPartitioningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graphutils=" << algorithm << " ";
  sequential.print(out, prefix + "sequential.");
}

void RefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graphutils=" << algorithm << " ";
  lp.print(out, prefix + "lp.");
}

void ParallelContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_threads=" << num_threads << " "                                         //
      << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
      << prefix << "mpi_thread_support=" << mpi_thread_support << " ";                          //
}

void PartitionContext::setup(const DistributedGraph &graph) {
  _global_n = graph.global_n();
  _global_m = graph.global_m();
  _global_total_node_weight = mpi::allreduce(graph.total_node_weight(), MPI_SUM, graph.communicator());
  _local_n = graph.n();
  _total_n = graph.total_n();
  _local_m = graph.m();
  _total_node_weight = graph.total_node_weight();

  setup_perfectly_balanced_block_weights();
  setup_max_block_weights();
}

void PartitionContext::setup_perfectly_balanced_block_weights() {
  _perfectly_balanced_block_weights.resize(k);

  const BlockWeight perfectly_balanced_block_weight = std::ceil(static_cast<double>(global_total_node_weight()) / k);
  tbb::parallel_for<BlockID>(
      0, k, [&](const BlockID b) { _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight; });
}

void PartitionContext::setup_max_block_weights() {
  _max_block_weights.resize(k);

  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    _max_block_weights[b] =
        static_cast<BlockWeight>((1.0 + epsilon) * static_cast<double>(perfectly_balanced_block_weight(b)));
  });
}

void PartitionContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "k=" << k << " "             //
      << prefix << "epsilon=" << epsilon << " " //
      << prefix << "mode=" << mode << " ";      //
}

void Context::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graph_filename=" << graph_filename << " "         //
      << prefix << "load_edge_balanced=" << load_edge_balanced << " " //
      << prefix << "seed=" << seed << " "                             //
      << prefix << "quiet=" << quiet << " ";                          //
  partition.print(out, prefix + "partition.");
  parallel.print(out, prefix + "parallel.");
  coarsening.print(out, prefix + "coarsening.");
  initial_partitioning.print(out, prefix + "initial_partitioning.");
  refinement.print(out, prefix + "refinement.");
}

std::ostream &operator<<(std::ostream &out, const Context &context) {
  context.print(out);
  return out;
}

Context create_default_context() {
  // clang-format off
  return {
    .graph_filename = "",
    .load_edge_balanced = false,
    .seed = 0,
    .quiet = false,
    .partition = {
      /* .k = */ 0,
      /* .epsilon = */ 0.03,
      /* .mode = */ PartitioningMode::KWAY,
    },
    .parallel = {
      .num_threads = 1,
      .use_interleaved_numa_allocation = true,
      .mpi_thread_support = MPI_THREAD_FUNNELED,
    },
    .coarsening = {
      .use_local_clustering = false,
      .use_global_clustering = true,
      .global_clustering_algorithm = GlobalClusteringAlgorithm::GLOBAL_LP,
      .global_contraction_algorithm = GlobalContractionAlgorithm::MINIMAL_MIGRATION,
      .contraction_limit = 5000,
      .local_lp = {
        .num_iterations = 1,
        .large_degree_threshold = 1'000'000,
        .max_num_neighbors = kInvalidNodeID,
        .merge_singleton_clusters = true,
        .merge_nonadjacent_clusters_threshold = 0.5,
        .total_num_chunks = 0, // unused
        .num_chunks = 0, // unused
        .min_num_chunks = 0, // unused
      },
      .global_lp = {
        .num_iterations = 5,
        .large_degree_threshold = 1'000'000,
        .max_num_neighbors = kInvalidNodeID,
        .merge_singleton_clusters = true,
        .merge_nonadjacent_clusters_threshold = 0.5,
        .total_num_chunks = 128,
        .num_chunks = 0,
        .min_num_chunks = 8,
      },
      .cluster_weight_limit = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
      .cluster_weight_multiplier = 1.0,
    },
    .initial_partitioning = {
      .algorithm = InitialPartitioningAlgorithm::KAMINPAR,
      .sequential = shm::create_default_context(),
    },
    .refinement = {
      .algorithm = KWayRefinementAlgorithm::PROB_LP,
      .lp = {
        .num_iterations = 5,
        .total_num_chunks = 128,
        .num_chunks = 0,
        .min_num_chunks = 8,
        .num_move_attempts = 2,
      }
    }
  };
  // clang-format on
}
} // namespace dkaminpar