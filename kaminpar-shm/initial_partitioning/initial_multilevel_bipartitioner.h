/*******************************************************************************
 * Simple interface for the initial (bi)partitioning code.
 *
 * Initial bipartitioning uses a sequential multilevel to compute high-quality
 * bipartitions.
 *
 * For coarsening, the code uses sequential label propagation, which is
 * interleaved with the construction of the next coarse graph. Bipartitioning
 * is done by a pool of simple algorithms (BFS, greedy graph growing, random).
 * Refinement is done by a 2-way sequential FM algorithm.
 *
 * Constructing an object of InitialPartitioner is relatively expensive;
 * especially if one wants to compute *many* bipartitions (i.e., if k is large).
 * Thus, objects should be kept in (thread-local) memory and be re-used to
 * compute multiple bipartitions (call init() for each new graph).
 *
 * Data structures are re-allocated to a larger size whenever necessary and never
 * shrink.
 *
 * @file:   initial_multilevel_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"
#include "kaminpar-shm/initial_partitioning/initial_pool_bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

struct InitialPartitionerTimings {
  std::uint64_t coarsening_ms = 0;
  std::uint64_t coarsening_misc_ms = 0;
  std::uint64_t coarsening_call_ms = 0;
  std::uint64_t uncoarsening_ms = 0;
  std::uint64_t bipartitioning_ms = 0;
  std::uint64_t total_ms = 0;
  std::uint64_t misc_ms = 0;
  InitialCoarsenerTimings coarsening{};

  InitialPartitionerTimings &operator+=(const InitialPartitionerTimings &other) {
    coarsening_ms += other.coarsening_ms;
    uncoarsening_ms += other.uncoarsening_ms;
    bipartitioning_ms += other.bipartitioning_ms;
    misc_ms += other.misc_ms;
    coarsening += other.coarsening;
    coarsening_misc_ms += other.coarsening_misc_ms;
    coarsening_call_ms += other.coarsening_call_ms;
    total_ms += other.total_ms;
    return *this;
  }
};

class InitialMultilevelBipartitioner {
public:
  explicit InitialMultilevelBipartitioner(const Context &ctx);

  /**
   * Initializes the bipartitioner for bipartitioning a block-induced subgraph.
   *
   * This function prepares the bipartitioner to bipartition a subgraph extracted from a specific
   * block `current_block` of an graph which is already partitioned into `current_k` blocks, where
   * `current_k < ctx.partition.k`. The maximum block weights for the bipartition are computed based
   * on the `ctx.partition` context.
   *
   * After initialization, the `partition()` method can be called to perform the bipartitioning.
   *
   * @param graph Subgraph extracted from a specific block of an already partitioned graph.
   * @param current_block Block ID of the block from which the subgraph was extracted.
   * @param current_k Number of blocks in the already partitioned graph.
   */
  void initialize(const CSRGraph &graph, BlockID current_block, BlockID current_k);

  /**
   * Bipartitions the graph initialized by `initialize()`.
   *
   * The maximum block weights of the bipartition are computed based on the `ctx.partition` context
   * and the information passed to `initialize()`.
   *
   * @return A partitioned graph with two blocks.
   */
  PartitionedCSRGraph partition(InitialPartitionerTimings *timings = nullptr);

private:
  const CSRGraph *coarsen(InitialPartitionerTimings *timings);
  PartitionedCSRGraph uncoarsen(PartitionedCSRGraph p_graph);

  const CSRGraph *_graph;
  PartitionContext _p_ctx;

  const Context &_ctx;
  const InitialPartitioningContext &_i_ctx;

  std::unique_ptr<InitialCoarsener> _coarsener;
  std::unique_ptr<InitialPoolBipartitioner> _bipartitioner;
  std::unique_ptr<InitialRefiner> _refiner;
};

} // namespace kaminpar::shm
