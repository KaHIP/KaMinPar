/*******************************************************************************
 * Distributed label propagation refiner that uses a graph coloring to avoid
 * move conflicts.
 *
 * @file:   clp_refiner.h
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/refiner.h"

#include "kaminpar-common/parallel/vector_ets.h"

namespace kaminpar::dist {
class ColoredLPRefinerFactory : public GlobalRefinerFactory {
public:
  ColoredLPRefinerFactory(const Context &ctx);

  ColoredLPRefinerFactory(const ColoredLPRefinerFactory &) = delete;
  ColoredLPRefinerFactory &operator=(const ColoredLPRefinerFactory &) = delete;

  ColoredLPRefinerFactory(ColoredLPRefinerFactory &&) noexcept = default;
  ColoredLPRefinerFactory &operator=(ColoredLPRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class ColoredLPRefiner : public GlobalRefiner {
  SET_STATISTICS_FROM_GLOBAL();
  SET_DEBUG(false);

  using BlockGainsContainer = typename parallel::vector_ets<EdgeWeight>::Container;

  struct MoveCandidate {
    NodeID local_seq;
    GlobalNodeID node;
    BlockID from;
    BlockID to;
    EdgeWeight gain;
    NodeWeight weight;
  };

  class GainStatistics {
  public:
    void initialize(ColorID num_colors);
    void record_gain(EdgeWeight gain, ColorID c);
    void summarize_by_size(const NoinitVector<NodeID> &color_sizes, MPI_Comm comm) const;

  private:
    std::vector<EdgeWeight> _gain_per_color;
  };

public:
  ColoredLPRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  ColoredLPRefiner(const ColoredLPRefiner &) = delete;
  ColoredLPRefiner &operator=(const ColoredLPRefiner &) = delete;

  ColoredLPRefiner(ColoredLPRefiner &&) noexcept = default;
  ColoredLPRefiner &operator=(ColoredLPRefiner &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  NodeID find_moves(ColorID c);
  NodeID perform_moves(ColorID c);
  NodeID perform_best_moves(ColorID c);
  NodeID perform_local_moves(ColorID c);
  NodeID perform_probabilistic_moves(ColorID c);
  NodeID try_probabilistic_moves(ColorID c, const BlockGainsContainer &block_gains);
  void synchronize_state(ColorID c);

  auto reduce_move_candidates(std::vector<MoveCandidate> &&candidates)
      -> std::vector<MoveCandidate>;
  auto reduce_move_candidates(std::vector<MoveCandidate> &&a, std::vector<MoveCandidate> &&b)
      -> std::vector<MoveCandidate>;

  void handle_node(NodeID u);
  void activate_neighbors(NodeID u);

  const Context &_input_ctx;
  const ColoredLabelPropagationRefinementContext &_ctx;

  const PartitionContext &_p_ctx;
  DistributedPartitionedGraph &_p_graph;

  NoinitVector<std::uint8_t> _color_blacklist;
  NoinitVector<ColorID> _color_sizes;
  NoinitVector<NodeID> _color_sorted_nodes;

  NoinitVector<BlockWeight> _block_weight_deltas;
  NoinitVector<EdgeWeight> _gains;
  NoinitVector<BlockID> _next_partition;

  NoinitVector<std::uint8_t> _is_active;

  GainStatistics _gain_statistics;
};
} // namespace kaminpar::dist
