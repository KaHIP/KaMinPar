/*******************************************************************************
 * Sequential 2-way FM refinement used during initial bipartitioning.
 *
 * @file:   initial_fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/refinement/initial_refiner.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

namespace fm {

struct SimpleStoppingPolicy {
  void init(const CSRGraph *graph);
  [[nodiscard]] bool should_stop(const InitialFMRefinementContext &fm_ctx);
  void reset();
  void update(EdgeWeight gain);

private:
  std::size_t _num_steps;
};

// "Adaptive" random walk stopping policy
// Implementation copied from: KaHyPar -> AdvancedRandomWalkModelStopsSearch,
// Copyright (C) Sebastian Schlag
struct AdaptiveStoppingPolicy {
  void init(const CSRGraph *graph);
  [[nodiscard]] bool should_stop(const InitialFMRefinementContext &fm_ctx);
  void reset();
  void update(EdgeWeight gain);

private:
  double _beta = 0.0;
  std::size_t _num_steps = 0;
  double _variance = 0.0;
  double _Mk = 0.0;
  double _MkMinus1 = 0.0;
  double _Sk = 0.0;
  double _SkMinus1 = 0.0;
};

struct MaxWeightSelectionPolicy;
struct MaxGainSelectionPolicy;
struct MaxOverloadSelectionPolicy;

struct BalancedMinCutAcceptancePolicy;

} // namespace fm

/*!
 * Sequential 2-way FM refinement with two priority queues, one for each block.
 *
 * @tparam QueueSelectionPolicy Policy to choose the queue from which to select the next node.
 * @tparam CutAcceptancePolicy Policty to decide which cuts to accept.
 * @tparam StoppingPolicy Policy to decide when to abort the local search.
 */
template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
class InitialFMRefiner : public InitialRefiner {
  static constexpr NodeID kChunkSize = 64;
  static constexpr std::size_t kNumberOfNodePermutations = 32;

public:
  explicit InitialFMRefiner(const InitialFMRefinementContext &r_ctx) : _r_ctx(r_ctx) {}

  void init(const CSRGraph &graph) final;

  bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  [[nodiscard]] bool abort(EdgeWeight prev_edge_weight, EdgeWeight cur_edge_weight) const;

  EdgeWeight round(PartitionedCSRGraph &p_graph);

  void init_pq(const PartitionedCSRGraph &p_graph);

  void insert_node(const PartitionedCSRGraph &p_graph, NodeID u);

  EdgeWeight compute_gain_from_scratch(const PartitionedCSRGraph &p_graph, NodeID u);

  void init_weighted_degrees();

  bool is_boundary_node(const PartitionedCSRGraph &p_graph, NodeID u);

  bool validate_pqs(const PartitionedCSRGraph &p_graph);

  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;
  const InitialFMRefinementContext &_r_ctx;

  std::array<BinaryMinHeap<EdgeWeight>, 2> _queues{
      BinaryMinHeap<EdgeWeight>{0}, BinaryMinHeap<EdgeWeight>{0}
  };
  Marker<> _marker{0};
  ScalableVector<EdgeWeight> _weighted_degrees{};
  std::vector<NodeID> _moves;

  StoppingPolicy _stopping_policy{};

  Random &_rand = Random::instance();

  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _permutations;
  std::vector<NodeID> _chunks;
};

using InitialSimple2WayFM = InitialFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::SimpleStoppingPolicy>;

using InitialAdaptive2WayFM = InitialFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::AdaptiveStoppingPolicy>;

} // namespace kaminpar::shm
