/***********************************************************************************************************************
 * @file:   colored_lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds
 *determined by a graph coloring.
 **********************************************************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/parallel/vector_ets.h"

namespace kaminpar::dist {
class ColoredLPRefiner : public Refiner {
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
  ColoredLPRefiner(const Context &ctx);

  ColoredLPRefiner(const ColoredLPRefiner &) = delete;
  ColoredLPRefiner &operator=(const ColoredLPRefiner &) = delete;
  ColoredLPRefiner(ColoredLPRefiner &&) noexcept = default;
  ColoredLPRefiner &operator=(ColoredLPRefiner &&) = delete;

  void initialize(const DistributedGraph &graph) final;
  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

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

  const PartitionContext *_p_ctx;
  DistributedPartitionedGraph *_p_graph;

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
