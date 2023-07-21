/*******************************************************************************
 * Distributed FM refiner.
 *
 * @file:   local_fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.08.2022
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/logger.h"
#include "common/parallel/atomic.h"

namespace kaminpar::dist {
class LocalFMRefinerFactory : public GlobalRefinerFactory {
public:
  LocalFMRefinerFactory(const Context &ctx);

  LocalFMRefinerFactory(const LocalFMRefinerFactory &) = delete;
  LocalFMRefinerFactory &operator=(const LocalFMRefinerFactory &) = delete;

  LocalFMRefinerFactory(LocalFMRefinerFactory &&) noexcept = default;
  LocalFMRefinerFactory &operator=(LocalFMRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class LocalFMRefiner : public GlobalRefiner {
  SET_STATISTICS(true);

  struct Statistics {
    // Sizes of search graphs
    tbb::concurrent_vector<NodeID> graphs_n{};
    tbb::concurrent_vector<EdgeID> graphs_m{};
    tbb::concurrent_vector<NodeID> graphs_border_n{};

    // Number of move conflicts when applying moves from search graphs to the
    // global partition
    parallel::Atomic<NodeID> num_conflicts{0};

    // Improvement statistics
    parallel::Atomic<NodeID> num_searches_with_improvement{0};
    EdgeWeight initial_cut{kInvalidEdgeWeight};
    EdgeWeight final_cut{kInvalidEdgeWeight};

    void print() const;
  };

public:
  LocalFMRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  LocalFMRefiner(const LocalFMRefiner &) = delete;
  LocalFMRefiner &operator=(const LocalFMRefiner &) = delete;
  LocalFMRefiner(LocalFMRefiner &&) = default;
  LocalFMRefiner &operator=(LocalFMRefiner &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  void refinement_round();
  tbb::concurrent_vector<NodeID> find_seed_nodes();

  void build_local_graph(
      const NodeID seed,
      shm::Graph &out_graph,
      shm::PartitionedGraph &out_p_graph,
      std::vector<GlobalNodeID> &mapping,
      std::vector<bool> &fixed
  );

  void init_external_degrees();

  EdgeWeight &external_degree(const NodeID u, const BlockID b) {
    KASSERT(_external_degrees.size() >= _p_graph.n() * _p_graph.k());
    return _external_degrees[u * _p_graph.k() + b];
  }

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  // initialized by ctor
  const FMRefinementContext &_fm_ctx;

  // initalized by refine()
  std::vector<EdgeWeight> _external_degrees;

  // initialized here
  std::size_t _round{0};
  std::vector<parallel::Atomic<std::uint8_t>> _locked;

  Statistics _stats;
};
} // namespace kaminpar::dist
