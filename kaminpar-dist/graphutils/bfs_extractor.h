/*******************************************************************************
 * Algorithm to extract a region of the graph using a BFS search.
 *
 * @file:   bfs_extractor.h
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 ******************************************************************************/
#pragma once

#include <limits>
#include <type_traits>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/datastructures/preallocated_vector.h"

namespace kaminpar::dist::graph {
class BfsExtractor {
public:
  struct Result {
    std::unique_ptr<shm::Graph> graph;
    std::unique_ptr<shm::PartitionedGraph> p_graph;
    NoinitVector<GlobalNodeID> node_mapping;
  };

  enum HighDegreeStrategy {
    IGNORE,
    TAKE_ALL,
    SAMPLE,
    CUT
  };

  enum ExteriorStrategy {
    EXCLUDE,
    INCLUDE,
    CONTRACT
  };

  BfsExtractor(const DistributedGraph &graph);

  BfsExtractor &operator=(const BfsExtractor &) = delete;
  BfsExtractor(const BfsExtractor &) = delete;

  BfsExtractor &operator=(BfsExtractor &&) = delete;
  BfsExtractor(BfsExtractor &&) noexcept = default;

  void initialize(const DistributedPartitionedGraph &p_graph);

  Result extract(const std::vector<NodeID> &seed_nodes);

  void set_max_hops(PEID max_hops);
  void set_max_radius(GlobalNodeID max_radius);
  void set_high_degree_threshold(GlobalEdgeID high_degree_threshold);
  void set_high_degree_strategy(HighDegreeStrategy high_degree_strategy);
  void set_exterior_strategy(ExteriorStrategy exterior_strategy);

private:
  using GhostSeedNode = std::tuple<NodeID, NodeID>;         // distance, node
  using GhostSeedEdge = std::tuple<NodeID, NodeID, NodeID>; // distance, from, to
  using ExploredNode = std::pair<NodeID, bool>;             // is border node, node

  struct GraphFragment {
    NoinitVector<EdgeID> nodes;
    NoinitVector<GlobalNodeID> edges;
    NoinitVector<NodeWeight> node_weights;
    NoinitVector<EdgeWeight> edge_weights;
    NoinitVector<GlobalNodeID> node_mapping;
    NoinitVector<BlockID> partition;
  };

  struct ExploredSubgraph {
    NoinitVector<GhostSeedEdge> explored_ghosts;

    NoinitVector<EdgeID> nodes;
    NoinitVector<GlobalNodeID> edges;
    NoinitVector<NodeWeight> node_weights;
    NoinitVector<EdgeWeight> edge_weights;
    NoinitVector<GlobalNodeID> node_mapping;
    NoinitVector<BlockID> partition;

    GraphFragment build_fragment() {
      return {
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          std::move(node_mapping),
          std::move(partition)};
    }
  };

  ExploredSubgraph
  bfs(PEID current_hop,
      NoinitVector<GhostSeedNode> &seed_nodes,
      const NoinitVector<NodeID> &ignored_nodes);

  template <typename Lambda> void explore_outgoing_edges(NodeID node, Lambda &&action);

  std::pair<std::vector<NoinitVector<GhostSeedNode>>, std::vector<NoinitVector<NodeID>>>
  exchange_ghost_seed_nodes(std::vector<NoinitVector<GhostSeedEdge>> &next_ghost_seed_nodes);

  std::vector<GraphFragment>
  exchange_explored_subgraphs(const std::vector<ExploredSubgraph> &explored_subgraphs);

  Result combine_fragments(tbb::concurrent_vector<GraphFragment> &fragments);

  void init_external_degrees();
  EdgeWeight &external_degree(NodeID u, BlockID b);

  GlobalNodeID map_block_to_pseudo_node(BlockID block);
  BlockID map_pseudo_node_to_block(GlobalNodeID node);
  bool is_pseudo_block_node(GlobalNodeID node);

  //
  // Members
  //

  const DistributedGraph *_graph = nullptr;
  const DistributedPartitionedGraph *_p_graph = nullptr;

  PEID _max_hops = std::numeric_limits<PEID>::max();
  GlobalNodeID _max_radius = std::numeric_limits<GlobalNodeID>::max();
  GlobalEdgeID _high_degree_threshold = std::numeric_limits<GlobalEdgeID>::max();
  HighDegreeStrategy _high_degree_strategy = HighDegreeStrategy::TAKE_ALL;
  ExteriorStrategy _exterior_strategy = ExteriorStrategy::EXCLUDE;

  NoinitVector<EdgeWeight> _external_degrees;

  Marker<> _finished_pe_search{
      static_cast<std::size_t>(mpi::get_comm_size(_graph->communicator()))};

  tbb::enumerable_thread_specific<Marker<>> _taken_ets{[&] {
    return Marker<>(_graph->total_n());
  }};
  tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> _external_degrees_ets{[&] {
    return FastResetArray<EdgeWeight>(_p_graph->k());
  }};
};
} // namespace kaminpar::dist::graph
