/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#pragma once

#include <cmath>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/refiner.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/binary_heap.h"

namespace kaminpar::shm {
namespace fm {
struct SharedData;
}

class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);
  ~FMRefiner(); // Required for the std::unique_ptr<> member.

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const KwayFMRefinementContext *_fm_ctx;
  std::unique_ptr<fm::SharedData> _shared;
};

class LocalizedFMRefiner {
public:
  LocalizedFMRefiner(
      int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      fm::SharedData &shared
  );

  EdgeWeight run_batch();

private:
  template <typename PartitionedGraphType, typename GainCacheType>
  void insert_into_node_pq(
      const PartitionedGraphType &p_graph,
      const GainCacheType &gain_cache,
      NodeID u
  );

  void update_after_move(
      NodeID node, NodeID moved_node, BlockID moved_from, BlockID moved_to
  );

  template <typename PartitionedGraphType, typename GainCacheType>
  std::pair<BlockID, EdgeWeight> best_gain(
      const PartitionedGraphType &p_graph,
      const GainCacheType &gain_cache,
      NodeID u
  );

  bool update_block_pq();

  // Unique worker ID
  int _id;

  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  // Graph to work on
  PartitionedGraph &_p_graph;

  // Data shared among all workers
  fm::SharedData &_shared;

  // Data local to this worker
  DeltaPartitionedGraph _d_graph;                        // O(|Delta|) space
  DeltaGainCache<DenseGainCache> _d_gain_cache;          // O(|Delta|) space
  BinaryMaxHeap<EdgeWeight> _block_pq;                   // O(k) space
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pq; // O(k + |Touched|) space

  AdaptiveStoppingPolicy _stopping_policy;
};
} // namespace kaminpar::shm

