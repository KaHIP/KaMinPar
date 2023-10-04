/*******************************************************************************
 * Greedy balancing algorithm that moves clusters of nodes at a time.
 *
 * @file:   cluster_balancer.h
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-dist/refinement/balancer/clusters.h"
#include "kaminpar-dist/refinement/balancer/weight_buckets.h"
#include "kaminpar-dist/refinement/refiner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
struct ClusterBalancerMemoryContext;

class ClusterBalancerFactory : public GlobalRefinerFactory {
public:
  ClusterBalancerFactory(const Context &ctx);

  ~ClusterBalancerFactory();

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void take_m_ctx(ClusterBalancerMemoryContext m_ctx);

private:
  const Context &_ctx;

  std::unique_ptr<ClusterBalancerMemoryContext> _m_ctx;
};

class ClusterBalancer : public GlobalRefiner {
  struct ClusterStatistics {
    NodeID cluster_count = 0;
    NodeID node_count = 0;
    NodeID min_cluster_size = 0;
    NodeID max_cluster_size = 0;
  };

  struct Statistics {
    int num_rounds = 0;
    std::vector<ClusterStatistics> cluster_stats;

    int num_seq_rounds = 0;
    int num_seq_cluster_moves = 0;
    int num_seq_node_moves = 0;
    double seq_imbalance_reduction = 0.0;
    EdgeWeight seq_cut_increase = 0;

    int num_par_rounds = 0;
    int num_par_cluster_moves = 0;
    int num_par_node_moves = 0;
    int num_par_dicing_attempts = 0;
    int num_par_balanced_moves = 0;
    double par_imbalance_reduction = 0.0;
    EdgeWeight par_cut_increase = 0;

    void reset();
    void print();
  };

public:
  ClusterBalancer(
      ClusterBalancerFactory &factory,
      const Context &ctx,
      DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      ClusterBalancerMemoryContext m_ctx
  );

  ClusterBalancer(const ClusterBalancer &) = delete;
  ClusterBalancer &operator=(const ClusterBalancer &) = delete;

  ClusterBalancer(ClusterBalancer &&) = delete;
  ClusterBalancer &operator=(ClusterBalancer &&) = delete;

  ~ClusterBalancer();

  operator ClusterBalancerMemoryContext() &&;

  void initialize() final;
  bool refine() final;

private:
  void rebuild_clusters();
  void init_clusters();
  void clear();

  void try_pq_insertion(NodeID set);
  void try_pq_update(NodeID set);

  void perform_parallel_round();

  bool use_sequential_rebalancing() const;
  bool use_parallel_rebalancing() const;
  ClusterStrategy get_cluster_strategy() const;

  struct MoveCandidate {
    PEID owner;
    NodeID id;
    NodeWeight weight;
    double gain;
    BlockID from;
    BlockID to;
  };

  void perform_sequential_round();
  std::vector<MoveCandidate> pick_sequential_candidates();
  void perform_moves(const std::vector<MoveCandidate> &candidates, bool update_block_weights);

  BlockWeight overload(BlockID block) const;
  BlockWeight underload(BlockID block) const;
  bool is_overloaded(BlockID block) const;
  BlockID count_overloaded_blocks() const;

  bool assign_feasible_target_block(
      MoveCandidate &candidate, const std::vector<BlockWeight> &deltas
  ) const;

  NodeWeight compute_cluster_weight_limit() const;

  std::string dbg_get_partition_state_str() const;
  std::string dbg_get_pq_state_str() const;
  bool dbg_validate_pq_weights() const;
  bool dbg_validate_bucket_weights() const;
  bool dbg_validate_cluster_conns() const;
  NodeID dbg_count_nodes_in_clusters(const std::vector<MoveCandidate> &candidates) const;

  Random &_rand = Random::instance();

  ClusterBalancerFactory &_factory;
  const Context &_ctx;
  const ClusterBalancerContext &_cb_ctx;
  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  DynamicBinaryMinMaxForest<NodeID, double> _pqs;
  NoinitVector<BlockWeight> _pq_weights;
  Marker<> _moved_marker;

  Buckets _weight_buckets;
  Clusters _clusters;

  double _current_parallel_rebalance_fraction;

  bool _stalled = false;

  Statistics _stats;
};
} // namespace kaminpar::dist
