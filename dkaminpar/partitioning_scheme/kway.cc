/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "dkaminpar/partitioning_scheme/kway.h"

#include "dkaminpar/algorithm/allgather_graph.h"
#include "dkaminpar/algorithm/distributed_graph_contraction.h"
#include "dkaminpar/coarsening/distributed_local_label_propagation_clustering.h"
#include "dkaminpar/refinement/distributed_probabilistic_label_propagation_refiner.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "kaminpar/metrics.h"
#include "kaminpar/partitioning_scheme/partitioning.h"

namespace dkaminpar {
SET_DEBUG(true);

KWayPartitioningScheme::KWayPartitioningScheme(const DistributedGraph &graph, const Context &ctx)
    : _graph{graph},
      _ctx{ctx} {}

DistributedPartitionedGraph KWayPartitioningScheme::partition() {
  // Coarsen graph
  std::vector<DistributedGraph> graph_hierarchy;
  std::vector<scalable_vector<NodeID>> mapping_hierarchy;

  const DistributedGraph *c_graph = &_graph;
  while (c_graph->n() > 2 * 160) {
    DBG << "... lp";
    const NodeWeight
        max_cluster_weight = shm::compute_max_cluster_weight(c_graph->global_n(), c_graph->total_node_weight(),
                                                             _ctx.initial_partitioning.sequential.partition,
                                                             _ctx.initial_partitioning.sequential.coarsening);
    // TODO total_n() only required for rating map
    DistributedLocalLabelPropagationClustering coarsener(c_graph->total_n(), _ctx.coarsening);
    auto &clustering = coarsener.compute_clustering(*c_graph, max_cluster_weight);
    MPI_Barrier(MPI_COMM_WORLD);
    DBG << "... contract";
    auto [contracted_graph, mapping, mem] = graph::contract_local_clustering(*c_graph, clustering);
    DBG << ".... ok";
    MPI_Barrier(MPI_COMM_WORLD);
    graph::debug::validate(contracted_graph);
    const bool converged = contracted_graph.global_n() == c_graph->global_n();
    graph_hierarchy.push_back(std::move(contracted_graph));
    mapping_hierarchy.push_back(std::move(mapping));
    c_graph = &graph_hierarchy.back();

    LOG << "=> n=" << c_graph->global_n() << " m=" << c_graph->global_m()
        << " max_node_weight=" << c_graph->max_node_weight() << " max_cluster_weight=" << max_cluster_weight;
    if (converged) {
      LOG << "==> Coarsening converged";
      break;
    }
  }

  // initial partitioning
  auto shm_graph = graph::allgather(*c_graph);
  auto shm_ctx = _ctx.initial_partitioning.sequential;
  shm_ctx.refinement.lp.num_iterations = 1;
  shm_ctx.partition.k = _ctx.partition.k;
  shm_ctx.partition.epsilon = _ctx.partition.epsilon;
  shm_ctx.setup(shm_graph);

  shm::Logger::set_quiet_mode(true);
  auto shm_p_graph = shm::partitioning::partition(shm_graph, shm_ctx);
  shm::Logger::set_quiet_mode(_ctx.quiet);
  DLOG << "Obtained " << shm_ctx.partition.k << "-way partition with cut=" << shm::metrics::edge_cut(shm_p_graph)
       << " and imbalance=" << shm::metrics::imbalance(shm_p_graph);

  DistributedPartitionedGraph dist_p_graph = graph::reduce_scatter(*c_graph, std::move(shm_p_graph));
  graph::debug::validate_partition(dist_p_graph);

  DLOG << "Initial partition: cut=" << metrics::edge_cut(dist_p_graph)
       << " imbalance=" << metrics::imbalance(dist_p_graph);

  auto refine = [&](DistributedPartitionedGraph &p_graph) {
    if (_ctx.refinement.algorithm == KWayRefinementAlgorithm::NOOP) { return; }
    DBG << "create local_n=" << _ctx.partition.local_n() << " k=" << _ctx.partition.k;
    DistributedProbabilisticLabelPropagationRefiner refiner(_ctx);
    DBG << "init";
    refiner.initialize(p_graph.graph(), _ctx.partition);

    for (std::size_t i = 0; i < _ctx.refinement.lp.num_iterations; ++i) {
      DBG << "iter " << i;
      refiner.refine(p_graph);
      DBG << "validate";
      graph::debug::validate_partition(p_graph);
    }
  };

  // Uncoarsen and refine
  DBG << "Calling refiner";
  refine(dist_p_graph);
  while (!graph_hierarchy.empty()) {
    // (1) Uncoarsen graph
    auto mapping = std::move(mapping_hierarchy.back());
    graph_hierarchy.pop_back();
    mapping_hierarchy.pop_back(); // destroy graph wrapped in dist_p_graph, but partition access is still ok

    // create partition for new coarsest graph
    const auto *current_graph = graph_hierarchy.empty() ? &_graph : &graph_hierarchy.back();
    scalable_vector<BlockID> partition(current_graph->total_n());
    tbb::parallel_for(static_cast<NodeID>(0), current_graph->total_n(),
                      [&](const NodeID u) { partition[u] = dist_p_graph.block(mapping[u]); });
    dist_p_graph = DistributedPartitionedGraph{current_graph, _ctx.partition.k, std::move(partition),
                                               std::move(dist_p_graph.take_block_weights())};

    // (2) Refine
    refine(dist_p_graph);

    DLOG << "Cut after LP: cut=" << metrics::edge_cut(dist_p_graph)
         << " imbalance=" << metrics::imbalance(dist_p_graph);
  }

  DLOG << "Done";
  return dist_p_graph;
}
} // namespace dkaminpar