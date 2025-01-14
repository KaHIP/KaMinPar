/*******************************************************************************
 * Context struct for the distributed graph partitioner.
 *
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#include "kaminpar-dist/context.h"

#include <algorithm>
#include <numeric>

#include <tbb/parallel_for.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"

namespace kaminpar::dist {

using namespace std::string_literals;

PartitionContext::PartitionContext(
    const BlockID k, const BlockID initial_k, const BlockID extension_k, const double epsilon
)
    : k(k),
      initial_k(initial_k),
      extension_k(extension_k),
      epsilon(epsilon) {}

PartitionContext::PartitionContext(const PartitionContext &other)
    : k(other.k),
      initial_k(other.initial_k),
      extension_k(other.extension_k),
      epsilon(other.epsilon),
      graph(other.graph == nullptr ? nullptr : std::make_unique<GraphContext>(*other.graph)) {}

PartitionContext &PartitionContext::operator=(const PartitionContext &other) {
  k = other.k;
  initial_k = other.initial_k;
  extension_k = other.extension_k;
  epsilon = other.epsilon;
  graph = other.graph == nullptr ? nullptr : std::make_unique<GraphContext>(*other.graph);
  return *this;
}

PartitionContext::~PartitionContext() = default;

GraphContext::GraphContext(const DistributedGraph &graph, const PartitionContext &p_ctx)
    : global_n(graph.global_n()),
      n(graph.n()),
      total_n(graph.total_n()),
      global_m(graph.global_m()),
      m(graph.m()),
      global_total_node_weight(graph.global_total_node_weight()),
      total_node_weight(graph.total_node_weight()),
      global_max_node_weight(graph.global_max_node_weight()),
      global_total_edge_weight(graph.global_total_edge_weight()),
      total_edge_weight(graph.total_edge_weight()) {
  setup_perfectly_balanced_block_weights(p_ctx.k);
  setup_max_block_weights(p_ctx.k, p_ctx.epsilon);
}

GraphContext::GraphContext(const shm::Graph &graph, const PartitionContext &p_ctx)
    : global_n(graph.n()),
      n(graph.n()),
      total_n(graph.n()),
      global_m(graph.m()),
      m(graph.m()),
      global_total_node_weight(graph.total_node_weight()),
      total_node_weight(graph.total_node_weight()),
      global_max_node_weight(graph.max_node_weight()),
      global_total_edge_weight(graph.total_edge_weight()),
      total_edge_weight(graph.total_edge_weight()) {
  setup_perfectly_balanced_block_weights(p_ctx.k);
  setup_max_block_weights(p_ctx.k, p_ctx.epsilon);
}

void GraphContext::setup_perfectly_balanced_block_weights(const BlockID k) {
  perfectly_balanced_block_weights.resize(k);

  const BlockWeight perfectly_balanced_block_weight = std::ceil(1.0 * global_total_node_weight / k);
  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;
  });
}

void GraphContext::setup_max_block_weights(const BlockID k, const double epsilon) {
  max_block_weights.resize(k);

  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    const BlockWeight max_eps_weight = static_cast<BlockWeight>(
        (1.0 + epsilon) * static_cast<double>(perfectly_balanced_block_weights[b])
    );
    const BlockWeight max_abs_weight = perfectly_balanced_block_weights[b] + global_max_node_weight;

    // Only relax weight on coarse levels
    if (static_cast<GlobalNodeWeight>(global_n) == global_total_node_weight) {
      max_block_weights[b] = max_eps_weight;
    } else {
      max_block_weights[b] = std::max(max_eps_weight, max_abs_weight);
    }
  });
}

int ChunksContext::compute(const ParallelContext &parallel) const {
  if (fixed_num_chunks > 0) {
    return fixed_num_chunks;
  }
  const PEID num_pes =
      scale_chunks_with_threads ? parallel.num_threads * parallel.num_mpis : parallel.num_mpis;
  return std::max<std::size_t>(min_num_chunks, total_num_chunks / num_pes);
}

bool LabelPropagationCoarseningContext::should_merge_nonadjacent_clusters(
    const NodeID old_n, const NodeID new_n
) const {
  return (1.0 - 1.0 * new_n / old_n) <= merge_nonadjacent_clusters_threshold;
}

bool RefinementContext::includes_algorithm(const RefinementAlgorithm algorithm) const {
  return std::find(algorithms.begin(), algorithms.end(), algorithm) != algorithms.end();
}

void GraphCompressionContext::setup(const DistributedCompressedGraph &graph) {
  constexpr int kRoot = 0;
  const MPI_Comm comm = graph.communicator();
  const int rank = mpi::get_comm_rank(comm);

  compressed_graph_sizes =
      mpi::gather<std::size_t, std::vector<std::size_t>>(graph.memory_space(), kRoot, comm);
  uncompressed_graph_sizes = mpi::gather<std::size_t, std::vector<std::size_t>>(
      graph.uncompressed_memory_space(), kRoot, comm
  );
  num_nodes = mpi::gather<NodeID, std::vector<NodeID>>(graph.n(), kRoot, comm);
  num_edges = mpi::gather<EdgeID, std::vector<EdgeID>>(graph.m(), kRoot, comm);

  const auto compression_ratios = mpi::gather(graph.compression_ratio(), kRoot, comm);
  if (rank == kRoot) {
    const auto size = static_cast<double>(compression_ratios.size());
    avg_compression_ratio =
        std::reduce(compression_ratios.begin(), compression_ratios.end()) / size;
    min_compression_ratio = *std::min_element(compression_ratios.begin(), compression_ratios.end());
    max_compression_ratio = *std::max_element(compression_ratios.begin(), compression_ratios.end());

    const auto largest_compressed_graph_it =
        std::max_element(compressed_graph_sizes.begin(), compressed_graph_sizes.end());
    largest_compressed_graph = *largest_compressed_graph_it;

    const auto largest_compressed_graph_rank =
        std::distance(compressed_graph_sizes.begin(), largest_compressed_graph_it);
    largest_compressed_graph_prev_size =
        largest_compressed_graph * compression_ratios[largest_compressed_graph_rank];

    const auto largest_uncompressed_graph_it =
        std::max_element(uncompressed_graph_sizes.begin(), uncompressed_graph_sizes.end());
    largest_uncompressed_graph = *largest_uncompressed_graph_it;

    const auto largest_uncompressed_graph_rank =
        std::distance(uncompressed_graph_sizes.begin(), largest_uncompressed_graph_it);
    largest_uncompressed_graph_after_size =
        largest_uncompressed_graph / compression_ratios[largest_uncompressed_graph_rank];
  }
}

} // namespace kaminpar::dist
