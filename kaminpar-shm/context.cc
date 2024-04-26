/*******************************************************************************
 * Context struct for KaMinPar.
 *
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/context.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/partition_utils.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

void GraphCompressionContext::setup(const Graph &graph) {
  high_degree_encoding = CompressedGraph::kHighDegreeEncoding;
  high_degree_threshold = CompressedGraph::kHighDegreeThreshold;
  high_degree_part_length = CompressedGraph::kHighDegreePartLength;
  interval_encoding = CompressedGraph::kIntervalEncoding;
  interval_length_treshold = CompressedGraph::kIntervalLengthTreshold;
  run_length_encoding = CompressedGraph::kRunLengthEncoding;
  stream_encoding = CompressedGraph::kStreamEncoding;
  isolated_nodes_separation = CompressedGraph::kIsolatedNodesSeparation;

  if (enabled) {
    if (const auto *compressed_graph =
            dynamic_cast<const CompressedGraph *>(graph.underlying_graph());
        compressed_graph != nullptr) {
      dismissed = false;
      compression_ratio = compressed_graph->compression_ratio();
      size_reduction = compressed_graph->size_reduction();
      num_high_degree_nodes = compressed_graph->num_high_degree_nodes();
      num_high_degree_parts = compressed_graph->num_high_degree_parts();
      num_interval_nodes = compressed_graph->num_interval_nodes();
      num_intervals = compressed_graph->num_intervals();
    } else {
      dismissed = true;
    }
  }
}

//
// PartitionContext
//

void PartitionContext::setup(const AbstractGraph &graph) {
  n = graph.n();
  m = graph.m();
  total_node_weight = graph.total_node_weight();
  total_edge_weight = graph.total_edge_weight();
  max_node_weight = graph.max_node_weight();
  setup_block_weights();
}

void PartitionContext::setup_block_weights() {
  block_weights.setup(*this);
}

//
// BlockWeightsContext
//

void BlockWeightsContext::setup(const PartitionContext &p_ctx) {
  KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
  KASSERT(p_ctx.k != 0u, "PartitionContext::k not initialized");
  KASSERT(
      p_ctx.total_node_weight != kInvalidNodeWeight,
      "PartitionContext::total_node_weight not initialized"
  );
  KASSERT(
      p_ctx.max_node_weight != kInvalidNodeWeight,
      "PartitionContext::max_node_weight not initialized"
  );

  const auto perfectly_balanced_block_weight =
      static_cast<NodeWeight>(std::ceil(1.0 * p_ctx.total_node_weight / p_ctx.k));
  const auto max_block_weight =
      static_cast<NodeWeight>((1.0 + p_ctx.epsilon) * perfectly_balanced_block_weight);

  _max_block_weights.resize(p_ctx.k);
  _perfectly_balanced_block_weights.resize(p_ctx.k);

  tbb::parallel_for<BlockID>(0, p_ctx.k, [&](const BlockID b) {
    _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;

    // relax balance constraint by max_node_weight on coarse levels only
    if (p_ctx.max_node_weight == 1) {
      _max_block_weights[b] = max_block_weight;
    } else {
      _max_block_weights[b] = std::max<NodeWeight>(
          max_block_weight, perfectly_balanced_block_weight + p_ctx.max_node_weight
      );
    }
  });
}

void BlockWeightsContext::setup(const PartitionContext &p_ctx, const BlockID input_k) {
  KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
  KASSERT(
      p_ctx.total_node_weight != kInvalidNodeWeight,
      "PartitionContext::total_node_weight not initialized"
  );
  KASSERT(
      p_ctx.max_node_weight != kInvalidNodeWeight,
      "PartitionContext::max_node_weight not initialized"
  );

  const double block_weight = 1.0 * p_ctx.total_node_weight / input_k;

  _max_block_weights.resize(p_ctx.k);
  _perfectly_balanced_block_weights.resize(p_ctx.k);

  tbb::parallel_for<BlockID>(0, p_ctx.k, [&](const BlockID b) {
    const BlockID final_k = compute_final_k(b, p_ctx.k, input_k);

    _perfectly_balanced_block_weights[b] = std::ceil(final_k * block_weight);

    const auto max_block_weight =
        static_cast<BlockWeight>((1.0 + p_ctx.epsilon) * _perfectly_balanced_block_weights[b]);

    // Relax balance constraint by max_node_weight on coarse levels only
    if (p_ctx.max_node_weight == 1) {
      _max_block_weights[b] = max_block_weight;
    } else {
      _max_block_weights[b] = std::max<BlockWeight>(
          max_block_weight, _perfectly_balanced_block_weights[b] + p_ctx.max_node_weight
      );
    }
  });
}

[[nodiscard]] const std::vector<BlockWeight> &BlockWeightsContext::all_max() const {
  return _max_block_weights;
}

[[nodiscard]] const std::vector<BlockWeight> &BlockWeightsContext::all_perfectly_balanced() const {
  return _perfectly_balanced_block_weights;
}

void Context::setup(const Graph &graph) {
  compression.setup(graph);
  partition.setup(graph);
}
} // namespace kaminpar::shm
