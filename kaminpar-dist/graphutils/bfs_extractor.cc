/*******************************************************************************
 * Algorithm to extract a region of the graph using a BFS search.
 *
 * @file:   bfs_extractor.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 ******************************************************************************/
#include "kaminpar-dist/graphutils/bfs_extractor.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <random>
#include <vector>

#include <tbb/parallel_for.h>

#include "kaminpar-mpi/sparse_alltoall.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/datastructures/growt.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist::graph {
SET_DEBUG(false);

BfsExtractor::BfsExtractor(const DistributedGraph &graph) : _graph(&graph) {}

void BfsExtractor::initialize(const DistributedPartitionedGraph &p_graph) {
  _p_graph = &p_graph;
}

auto BfsExtractor::extract(const std::vector<NodeID> &seed_nodes) -> Result {
  SCOPED_TIMER("BFS extraction");

  // Initialize external degrees if needed
  if (_exterior_strategy == ExteriorStrategy::CONTRACT && _external_degrees.empty()) {
    init_external_degrees();
  }

  const PEID size = mpi::get_comm_size(_graph->communicator());
  const PEID rank = mpi::get_comm_rank(_graph->communicator());

  NoinitVector<GhostSeedNode> initial_seed_nodes;
  for (const auto seed_node : seed_nodes) {
    initial_seed_nodes.emplace_back(0, seed_node);
  }

  DBG << "Running initial BFS on local PE ...";
  START_TIMER("Initial BFS");
  auto local_bfs_result = bfs(0, initial_seed_nodes, {});
  STOP_TIMER();

  std::vector<NoinitVector<GhostSeedEdge>> cur_ghost_seed_edges(size);
  cur_ghost_seed_edges[rank] = std::move(local_bfs_result.explored_ghosts);

  tbb::concurrent_vector<GraphFragment> fragments;
  fragments.push_back(local_bfs_result.build_fragment());

  std::vector<ExploredSubgraph> local_bfs_results(size);

  for (PEID hop = 0; hop < _max_hops; ++hop) {
    START_TIMER("Exchange ghost seed nodes");
    auto ghost_exchange_result = exchange_ghost_seed_nodes(cur_ghost_seed_edges);
    STOP_TIMER();

    auto &next_ghost_seed_nodes = ghost_exchange_result.first;
    auto &next_ignored_nodes = ghost_exchange_result.second;

    START_TIMER("Continued BFS");
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      DBG << "Continue BFS from PE " << pe << " with " << next_ghost_seed_nodes[pe].size()
          << " ghost nodes and " << next_ignored_nodes[pe].size() << " ignored nodes";
      local_bfs_results[pe] = bfs(hop + 1, next_ghost_seed_nodes[pe], next_ignored_nodes[pe]);
      cur_ghost_seed_edges[pe] = std::move(local_bfs_results[pe].explored_ghosts);
    });
    STOP_TIMER();

    START_TIMER("Exchange explored subgraphs");
    auto local_fragments = exchange_explored_subgraphs(local_bfs_results);
    STOP_TIMER();

    for (auto &fragment : local_fragments) {
      fragments.push_back(std::move(fragment));
    }
  }

  SCOPED_TIMER("Combine subgraph fragments");
  return combine_fragments(fragments);
}

auto BfsExtractor::exchange_ghost_seed_nodes(
    std::vector<NoinitVector<GhostSeedEdge>> &outgoing_ghost_seed_edges
) -> std::pair<std::vector<NoinitVector<GhostSeedNode>>, std::vector<NoinitVector<NodeID>>> {
  const PEID size = mpi::get_comm_size(_graph->communicator());

  // Exchange new ghost nodes
  std::vector<NoinitVector<NodeID>> sendbufs(size);
  for (PEID pe = 0; pe < size; ++pe) {
    // Ghost seed nodes to continue the BFS search initiated on PE `pe`
    const auto &ghost_seed_edges = outgoing_ghost_seed_edges[pe];

    for (const auto &[local_interface_node, local_ghost_node, distance] : ghost_seed_edges) {
      KASSERT(_graph->is_ghost_node(local_ghost_node));

      const auto ghost_owner = _graph->ghost_owner(local_ghost_node);
      const auto global_ghost_node = _graph->local_to_global_node(local_ghost_node);
      const auto remote_ghost_node =
          asserting_cast<NodeID>(global_ghost_node - _graph->node_distribution(ghost_owner));

      sendbufs[ghost_owner].push_back(static_cast<NodeID>(pe));
      sendbufs[ghost_owner].push_back(local_interface_node);
      sendbufs[ghost_owner].push_back(remote_ghost_node);
      sendbufs[ghost_owner].push_back(distance);
    }
  }

  std::vector<NoinitVector<GhostSeedEdge>> incoming_ghost_seed_edges(size);
  mpi::sparse_alltoall<NodeID>(
      std::move(sendbufs),
      [&](const auto &recvbuf, const PEID pe) {
        for (std::size_t i = 0; i < recvbuf.size();) {
          const PEID initiating_pe = asserting_cast<PEID>(recvbuf[i++]);
          const NodeID remote_ghost_node = recvbuf[i++];
          const NodeID local_interface_node = recvbuf[i++];
          const NodeID distance = recvbuf[i++];

          const auto global_ghost_node = _graph->node_distribution(pe) + remote_ghost_node;
          const auto local_ghost_node = _graph->global_to_local_node(global_ghost_node);

          incoming_ghost_seed_edges[initiating_pe].emplace_back(
              local_ghost_node, local_interface_node, distance
          );
        }
      },
      _graph->communicator()
  );

  // Filter edges that where already explored from this PE
  std::vector<NoinitVector<GhostSeedNode>> next_ghost_seed_nodes(size);
  std::vector<NoinitVector<NodeID>> next_ignored_nodes(size);

  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    auto &outgoing_edges = outgoing_ghost_seed_edges[pe];
    auto &incoming_edges = incoming_ghost_seed_edges[pe];
    std::sort(outgoing_edges.begin(), outgoing_edges.end());
    std::sort(incoming_edges.begin(), incoming_edges.end());

    // std::size_t outgoing_pos  = 0;
    for (std::size_t incoming_pos = 0; incoming_pos < incoming_edges.size(); ++incoming_pos) {
      const auto &[cur_ghost_node, cur_interface_node, distance] = incoming_edges[incoming_pos];

      // @todo filter
      // Only use this as a ghost node

      if (cur_interface_node == -1) {
        DBG0 << "received ghost seed node " << cur_interface_node << " with distance " << distance
             << ", comign from ghost node " << cur_ghost_node
             << " == " << _graph->local_to_global_node(cur_ghost_node) << " from PE " << pe;
      }

      next_ghost_seed_nodes[pe].emplace_back(distance, cur_interface_node);
      next_ignored_nodes[pe].push_back(cur_ghost_node);
    }
  });

  return {std::move(next_ghost_seed_nodes), std::move(next_ignored_nodes)};
}

auto BfsExtractor::exchange_explored_subgraphs(
    const std::vector<ExploredSubgraph> &explored_subgraphs
) -> std::vector<GraphFragment> {
  const PEID size = mpi::get_comm_size(_graph->communicator());

  // Preparse sendbuffers
  std::vector<NoinitVector<EdgeID>> nodes_sendbufs(size);
  std::vector<NoinitVector<GlobalNodeID>> edges_sendbufs(size);
  std::vector<NoinitVector<NodeWeight>> node_weights_sendbufs(size);
  std::vector<NoinitVector<EdgeWeight>> edge_weights_sendbufs(size);
  std::vector<NoinitVector<BlockID>> partition_sendbufs(size);
  std::vector<NoinitVector<GlobalNodeID>> node_mapping_sendbufs(size);

  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    nodes_sendbufs[pe] = std::move(explored_subgraphs[pe].nodes);
    edges_sendbufs[pe] = std::move(explored_subgraphs[pe].edges);
    node_weights_sendbufs[pe] = std::move(explored_subgraphs[pe].node_weights);
    edge_weights_sendbufs[pe] = std::move(explored_subgraphs[pe].edge_weights);
    partition_sendbufs[pe] = std::move(explored_subgraphs[pe].partition);
    node_mapping_sendbufs[pe] = std::move(explored_subgraphs[pe].node_mapping);
  });

  auto nodes_recvbufs =
      mpi::sparse_alltoall_get<EdgeID>(std::move(nodes_sendbufs), _graph->communicator());
  auto edges_recvbufs =
      mpi::sparse_alltoall_get<GlobalNodeID>(std::move(edges_sendbufs), _graph->communicator());
  auto node_weights_recvbufs = mpi::sparse_alltoall_get<NodeWeight>(
      std::move(node_weights_sendbufs), _graph->communicator()
  );
  auto edge_weights_recvbufs = mpi::sparse_alltoall_get<EdgeWeight>(
      std::move(edge_weights_sendbufs), _graph->communicator()
  );
  auto partition_recvbufs =
      mpi::sparse_alltoall_get<BlockID>(std::move(partition_sendbufs), _graph->communicator());
  auto node_mapping_recvbufs = mpi::sparse_alltoall_get<GlobalNodeID>(
      std::move(node_mapping_sendbufs), _graph->communicator()
  );

  std::vector<GraphFragment> fragments(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    fragments[pe] = {
        std::move(nodes_recvbufs[pe]),
        std::move(edges_recvbufs[pe]),
        std::move(node_weights_recvbufs[pe]),
        std::move(edge_weights_recvbufs[pe]),
        std::move(node_mapping_recvbufs[pe]),
        std::move(partition_recvbufs[pe])};
  });

  return fragments;
}

auto BfsExtractor::bfs(
    const PEID current_hop,
    NoinitVector<GhostSeedNode> &ghost_seed_nodes,
    const NoinitVector<NodeID> &ignored_nodes
) -> ExploredSubgraph {
  KASSERT([&] {
    for (const NodeID node : ignored_nodes) {
      KASSERT(
          !_graph->is_owned_node(node), "ignored nodes should be ghost nodes, not interface nodes"
      );
    }
    for (const auto &[distance, node] : ghost_seed_nodes) {
      KASSERT(_graph->is_owned_node(node), "seed nodes should be interface nodes, not ghost nodes");
    }
    return true;
  }());

  // Reset thread-local state from the previous BFS ...
  auto &taken = _taken_ets.local();
  auto &external_degrees_map = _external_degrees_ets.local();
  taken.reset();
  external_degrees_map.clear();

  // Prevent that we explore edges from which this BFS continues from another
  // PE
  for (const NodeID ignored_node : ignored_nodes) {
    taken.set(ignored_node);
    if (_graph->local_to_global_node(ignored_node) == 350099 ||
        _graph->local_to_global_node(ignored_node) == 1725765) {
      DBG0 << "ignored node " << _graph->local_to_global_node(ignored_node);
    }
  }

  // Initialize search from closest ghost seed nodes
  std::sort(ghost_seed_nodes.begin(), ghost_seed_nodes.end(), [](const auto &lhs, const auto &rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  });
  NodeID current_distance = (!ghost_seed_nodes.empty() ? std::get<0>(ghost_seed_nodes.front()) : 0);
  if (!ghost_seed_nodes.empty()) {
    DBG << "initial distance for continued search: " << current_distance << ", for node "
        << std::get<1>(ghost_seed_nodes.front());
  }

  std::queue<NodeID> front;
  NoinitVector<GhostSeedEdge> next_ghost_seed_edges;

  auto add_seed_nodes_to_search_front = [&](const NodeID with_distance) {
    if (with_distance > _max_radius) {
      return;
    }

    for (const auto &[distance, node] : ghost_seed_nodes) {
      if (distance == with_distance && !taken.get(node)) {
        if (_graph->local_to_global_node(node) == 350099 ||
            _graph->local_to_global_node(node) == 1725765) {
          DBG0 << "seed node " << node << " = " << _graph->local_to_global_node(node)
               << " with distance " << distance;
        }
        front.push(node);
        taken.set(node);
      } else if (with_distance < distance) {
        break;
      }
    }
  };

  // Initialize search front
  add_seed_nodes_to_search_front(current_distance);
  NodeID front_size = front.size();
  add_seed_nodes_to_search_front(current_distance + 1);

  NoinitVector<EdgeID> nodes;
  NoinitVector<GlobalNodeID> edges;
  NoinitVector<NodeWeight> node_weights;
  NoinitVector<EdgeWeight> edge_weights;
  NoinitVector<GlobalNodeID> node_mapping; // @todo makes explored_nodes redundant?
  NoinitVector<BlockID> partition;

  // Perform BFS
  while (!front.empty()) {
    const NodeID node = front.front();
    KASSERT(_graph->is_owned_node(node));
    front.pop();
    --front_size;
    if (_graph->local_to_global_node(node) == 350099 ||
        _graph->local_to_global_node(node) == 1725765) {
      DBG0 << "popped node " << node << " = " << _graph->local_to_global_node(node);
    }

    nodes.push_back(edges.size());
    node_weights.push_back(_graph->node_weight(node));
    node_mapping.push_back(_graph->local_to_global_node(node));
    partition.push_back(_p_graph->block(node));

    const bool is_distance_border_node = current_distance == _max_radius;
    const bool is_hop_border_node = current_hop == _max_hops;

    external_degrees_map.clear();

    explore_outgoing_edges(node, [&](const EdgeID edge, const NodeID neighbor) {
      const bool is_real_target =
          taken.get(neighbor) ||                                                                 //
          (_graph->is_owned_node(neighbor) && !is_distance_border_node) ||                       //
          (!_graph->is_owned_node(neighbor) && !is_distance_border_node && !is_hop_border_node); //

      if ((_graph->local_to_global_node(node) == 1725765 &&
           _graph->local_to_global_node(neighbor) == 350099) ||
          (_graph->local_to_global_node(neighbor) == 1725765 &&
           _graph->local_to_global_node(node) == 350099)) {
        DBG << "consider edge " << node << " = " << _graph->local_to_global_node(node) << " --> "
            << neighbor << " = " << _graph->local_to_global_node(neighbor)
            << "; is_real_target = " << is_real_target
            << "; is_distance_border_node = " << is_distance_border_node
            << "; is_hop_border_node = " << is_hop_border_node;
      }

      if (is_real_target) {
        edges.push_back(_graph->local_to_global_node(neighbor));
        edge_weights.push_back(_graph->edge_weight(edge));

        if (!taken.get(neighbor)) {
          taken.set(neighbor);

          if (_graph->local_to_global_node(neighbor) == 350099) {
            DBG0 << "took node " << neighbor << " = " << _graph->local_to_global_node(neighbor)
                 << ", while iterating over neighbors of node " << node << " with distance "
                 << current_distance;
          }

          if (_graph->is_owned_node(neighbor)) {
            front.push(neighbor);
          }
        }
        if (!_graph->is_owned_node(neighbor) && !is_hop_border_node) {
          if (_graph->local_to_global_node(neighbor) == 350099 ||
              _graph->local_to_global_node(neighbor) == 1725765) {
            DBG << "pushed node " << neighbor << " = " << _graph->local_to_global_node(neighbor)
                << ", while iterating over neighbors of node " << node << " with distance "
                << current_distance;
          }
          next_ghost_seed_edges.emplace_back(node, neighbor, current_distance + 1);
        }
      } else {
        external_degrees_map[_p_graph->block(neighbor)] += _graph->edge_weight(edge);
      }

      return true;
    });

    // Add edges to the pseudo-nodes representing the contracted exterior -- if there are any
    for (const auto &[block, weight] : external_degrees_map.entries()) {
      edges.push_back(map_block_to_pseudo_node(block));
      edge_weights.push_back(weight);
    }

    if (front_size == 0) {
      front_size = front.size();
      ++current_distance;
      add_seed_nodes_to_search_front(current_distance + 1);
    }
  }

  nodes.push_back(edges.size());

  ExploredSubgraph ans;
  ans.explored_ghosts = std::move(next_ghost_seed_edges);
  ans.nodes = std::move(nodes);
  ans.edges = std::move(edges);
  ans.node_weights = std::move(node_weights);
  ans.edge_weights = std::move(edge_weights);
  ans.node_mapping = std::move(node_mapping);
  ans.partition = std::move(partition);
  return ans;
}

template <typename Lambda>
void BfsExtractor::explore_outgoing_edges(const NodeID node, Lambda &&lambda) {
  const bool is_high_degree_node = _graph->degree(node) >= _high_degree_threshold;

  if (!is_high_degree_node || _high_degree_strategy == HighDegreeStrategy::TAKE_ALL) {
    for (const auto [e, v] : _graph->neighbors(node)) {
      if (!lambda(e, v)) {
        break;
      }
    }
  } else if (_high_degree_strategy == HighDegreeStrategy::CUT) {
    for (EdgeID e = _graph->first_edge(node); e < _graph->first_edge(node) + _high_degree_threshold;
         ++e) {
      if (!lambda(e, _graph->edge_target(e))) {
        break;
      }
    }
  } else if (_high_degree_strategy == HighDegreeStrategy::SAMPLE) {
    const double skip_prob = 1.0 * _high_degree_threshold / _graph->degree(node);
    std::geometric_distribution<EdgeID> skip_dist(skip_prob);

    for (EdgeID e = _graph->first_edge(node); e < _graph->first_invalid_edge(node);
         ++e) { // e += skip_dist(gen)) { // @todo
      if (!lambda(e, _graph->edge_target(e))) {
        break;
      }
    }
  } else {
    // do nothing for HighDegreeStrategy::IGNORE
  }
}

auto BfsExtractor::combine_fragments(tbb::concurrent_vector<GraphFragment> &fragments) -> Result {
  DBG << "Combining " << fragments.size() << " fragments to the resulting BFS search graph ...";
  for (const auto &fragment : fragments) {
    DBG << "  Fragment: n=" << fragment.nodes.size() << ", m=" << fragment.edges.size();
  }

  // Compute size of combined graph
  const NodeID real_n =
      parallel::accumulate(fragments.begin(), fragments.end(), 0, [&](const auto &fragment) {
        return fragment.nodes.size() - 1;
      });
  const NodeID pseudo_n = real_n + _p_graph->k(); // Include pseudo-nodes for contracted neighbors
  const EdgeID m =
      parallel::accumulate(fragments.begin(), fragments.end(), 0, [&](const auto &fragment) {
        return fragment.edges.size();
      });

  DBG << "Graph size: " << V(real_n) << V(pseudo_n) << V(m);

  // Allocate arrays for combined graph
  NoinitVector<GlobalNodeID> node_mapping(real_n);
  StaticArray<shm::EdgeID> nodes(pseudo_n + 1);
  StaticArray<shm::NodeID> edges(m);
  StaticArray<shm::NodeWeight> node_weights(pseudo_n);
  StaticArray<shm::EdgeWeight> edge_weights(m);
  StaticArray<shm::BlockID> partition(pseudo_n);

  // Compute node mapping
  std::vector<NodeID> first_node_id_for_fragment(fragments.size() + 1);
  std::vector<EdgeID> first_edge_id_for_fragment(fragments.size() + 1);
  tbb::parallel_for<std::size_t>(0, fragments.size(), [&](const std::size_t i) {
    first_node_id_for_fragment[i + 1] = fragments[i].nodes.size() - 1;
    first_edge_id_for_fragment[i + 1] = fragments[i].edges.size();
  });
  parallel::prefix_sum(
      first_node_id_for_fragment.begin(),
      first_node_id_for_fragment.end(),
      first_node_id_for_fragment.begin()
  );
  parallel::prefix_sum(
      first_edge_id_for_fragment.begin(),
      first_edge_id_for_fragment.end(),
      first_edge_id_for_fragment.begin()
  );

  growt::StaticGhostNodeMapping global_to_graph_mapping(first_node_id_for_fragment.back() + 1);
  tbb::parallel_for<std::size_t>(0, fragments.size(), [&](const std::size_t i) {
    for (std::size_t j = 0; j < fragments[i].node_mapping.size(); ++j) {
      const GlobalNodeID global_node = fragments[i].node_mapping[j];
      const NodeID new_node = first_node_id_for_fragment[i] + j;

      // New graph to global graph
      node_mapping[new_node] = global_node;

      // Global graph to new graph
      global_to_graph_mapping.insert(global_node + 1, new_node);
    }
  });

  // Construct graph arrays
  tbb::parallel_for<std::size_t>(0, fragments.size(), [&](const std::size_t i) {
    auto &fragment_nodes = fragments[i].nodes;
    auto &fragment_edges = fragments[i].edges;
    auto &fragment_node_weights = fragments[i].node_weights;
    auto &fragment_edge_weights = fragments[i].edge_weights;
    auto &fragment_partition = fragments[i].partition;

    EdgeID next_edge_id = first_edge_id_for_fragment[i];
    for (NodeID frag_u = 0; frag_u < fragment_nodes.size() - 1; ++frag_u) {
      const NodeID new_u = first_node_id_for_fragment[i] + frag_u;
      nodes[new_u] = next_edge_id;
      node_weights[new_u] = fragment_node_weights[frag_u];
      partition[new_u] = fragment_partition[frag_u];

      for (EdgeID frag_e = fragment_nodes[frag_u]; frag_e < fragment_nodes[frag_u + 1]; ++frag_e) {
        const EdgeID new_e = next_edge_id++;
        edge_weights[new_e] = fragment_edge_weights[frag_e];

        const NodeID global_v = fragment_edges[frag_e];
        if (is_pseudo_block_node(global_v)) {
          const BlockID block = map_pseudo_node_to_block(global_v);
          edges[new_e] = real_n + block;
        } else {
          auto it = global_to_graph_mapping.find(global_v + 1);
          KASSERT(
              it != global_to_graph_mapping.end(),
              "Could not find a mapping for global node " << global_v << ", contained in fragment "
                                                          << i << " of " << fragments.size()
          );
          edges[new_e] = (*it).second;

          if (((*it).second == 1321279 && new_u == 682490) ||
              ((*it).second == 682490 && new_u == 1321279)) {
            DBG << "Create edge " << new_u << " = " << node_mapping[new_u] << " -> " << (*it).second
                << " = " << global_v << " = " << node_mapping[(*it).second] << "; fragment " << i
                << " of " << fragments.size();
          }
        }
      }
    }

    if (i + 1 == fragments.size()) { // last fragment
      const NodeID last_u = first_node_id_for_fragment[i] + fragment_nodes.size() - 1;
      nodes[last_u] = next_edge_id;
    }
  });

  for (NodeID pseudo_node = real_n; pseudo_node < pseudo_n; ++pseudo_node) {
    partition[pseudo_node] = pseudo_node - real_n;
  }

  // Create nodes entries for pseudo-block nodes
  tbb::parallel_for<NodeID>(real_n, pseudo_n + 1, [&](const NodeID u) {
    nodes[u] = nodes[real_n];
  });

  // Construct shared-memory graph
  auto graph = std::make_unique<shm::Graph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
  );
  auto p_graph =
      std::make_unique<shm::PartitionedGraph>(*graph, _p_graph->k(), std::move(partition));

  mpi::barrier(_p_graph->communicator());
  return {std::move(graph), std::move(p_graph), std::move(node_mapping)};
}

void BfsExtractor::set_max_hops(const PEID max_hops) {
  _max_hops = max_hops;
}

void BfsExtractor::set_max_radius(const GlobalNodeID max_radius) {
  _max_radius = max_radius;
}

void BfsExtractor::set_high_degree_threshold(const GlobalEdgeID high_degree_threshold) {
  _high_degree_threshold = high_degree_threshold;
}

void BfsExtractor::set_high_degree_strategy(const HighDegreeStrategy high_degree_strategy) {
  _high_degree_strategy = high_degree_strategy;
}

void BfsExtractor::init_external_degrees() {
  _external_degrees.resize(_graph->n() * _p_graph->k());
  tbb::parallel_for<std::size_t>(0, _external_degrees.size(), [&](const std::size_t i) {
    _external_degrees[i] = 0;
  });

  _graph->pfor_nodes([&](const NodeID u) {
    for (const auto [e, v] : _graph->neighbors(u)) {
      const BlockID v_block = _p_graph->block(v);
      const EdgeWeight e_weight = _graph->edge_weight(e);
      external_degree(u, v_block) += e_weight;
    }
  });
}

EdgeWeight &BfsExtractor::external_degree(const NodeID u, const BlockID b) {
  KASSERT(u * _p_graph->k() + b < _external_degrees.size());
  return _external_degrees[u * _p_graph->k() + b];
}

GlobalNodeID BfsExtractor::map_block_to_pseudo_node(const BlockID block) {
  return _graph->global_n() + block;
}

BlockID BfsExtractor::map_pseudo_node_to_block(const GlobalNodeID node) {
  KASSERT(is_pseudo_block_node(node));
  return static_cast<BlockID>(node - _graph->global_n());
}

bool BfsExtractor::is_pseudo_block_node(const GlobalNodeID node) {
  return node >= _graph->global_n();
}
} // namespace kaminpar::dist::graph
