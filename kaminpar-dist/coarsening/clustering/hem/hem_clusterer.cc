/*******************************************************************************
 * Clusterer via heavy edge matching.
 *
 * @file:   hem_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#include "kaminpar-dist/coarsening/clustering/hem/hem_clusterer.h"

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(true);
}

HEMClusterer::HEMClusterer(const Context &ctx) : _input_ctx(ctx), _ctx(ctx.coarsening.hem) {}

void HEMClusterer::initialize(const DistributedGraph &graph) {
  mpi::barrier(graph.communicator());
  _graph = &graph;
  SCOPED_TIMER("Initialize HEM clustering");

  const auto coloring = [&] {
    // Graph is already sorted by a coloring -> reconstruct this coloring
    // @todo if we always want to do this, optimize this refiner
    if (graph.color_sorted()) {
      LOG << "Graph sorted by colors: using precomputed coloring";

      NoinitVector<ColorID> coloring(graph.n()
      ); // We do not actually need the colors for ghost nodes

      // @todo parallelize
      NodeID pos = 0;
      for (ColorID c = 0; c < graph.number_of_colors(); ++c) {
        const std::size_t size = graph.color_size(c);
        std::fill(coloring.begin() + pos, coloring.begin() + pos + size, c);
        pos += size;
      }

      return coloring;
    }

    // Otherwise, compute a coloring now
    LOG << "Computing new coloring";
    return compute_node_coloring_sequentially(graph, _ctx.chunks.compute(_input_ctx.parallel));
  }();

  const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
  const ColorID num_colors = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());

  TIMED_SCOPE("Allocation") {
    _color_sorted_nodes.resize(graph.n());
    _color_sizes.resize(num_colors + 1);
    _color_blacklist.resize(num_colors);
    tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
      _color_sorted_nodes[i] = 0;
    });
    tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) {
      _color_sizes[i] = 0;
    });
    tbb::parallel_for<std::size_t>(0, _color_blacklist.size(), [&](const std::size_t i) {
      _color_blacklist[i] = 0;
    });
  };

  TIMED_SCOPE("Count color sizes") {
    if (graph.color_sorted()) {
      const auto &color_sizes = graph.get_color_sizes();
      _color_sizes.assign(color_sizes.begin(), color_sizes.end());
    } else {
      graph.pfor_nodes([&](const NodeID u) {
        const ColorID c = coloring[u];
        KASSERT(c < num_colors);
        __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
      });
      parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
    }
  };

  TIMED_SCOPE("Sort nodes") {
    if (graph.color_sorted()) {
      // @todo parallelize
      std::iota(_color_sorted_nodes.begin(), _color_sorted_nodes.end(), 0);
    } else {
      graph.pfor_nodes([&](const NodeID u) {
        const ColorID c = coloring[u];
        const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
        KASSERT(i < _color_sorted_nodes.size());
        _color_sorted_nodes[i] = u;
      });
    }
  };

  TIMED_SCOPE("Compute color blacklist") {
    if (_ctx.small_color_blacklist == 0 ||
        (_ctx.only_blacklist_input_level && graph.global_n() != _input_ctx.partition.graph->global_n
        )) {
      return;
    }

    NoinitVector<GlobalNodeID> global_color_sizes(num_colors);
    tbb::parallel_for<ColorID>(0, num_colors, [&](const ColorID c) {
      global_color_sizes[c] = _color_sizes[c + 1] - _color_sizes[c];
    });
    MPI_Allreduce(
        MPI_IN_PLACE,
        global_color_sizes.data(),
        asserting_cast<int>(num_colors),
        mpi::type::get<GlobalNodeID>(),
        MPI_SUM,
        graph.communicator()
    );

    // @todo parallelize the rest of this section
    std::vector<ColorID> sorted_by_size(num_colors);
    std::iota(sorted_by_size.begin(), sorted_by_size.end(), 0);
    std::sort(
        sorted_by_size.begin(),
        sorted_by_size.end(),
        [&](const ColorID lhs, const ColorID rhs) {
          return global_color_sizes[lhs] < global_color_sizes[rhs];
        }
    );

    GlobalNodeID excluded_so_far = 0;
    for (const ColorID c : sorted_by_size) {
      excluded_so_far += global_color_sizes[c];
      const double percentage = 1.0 * excluded_so_far / graph.global_n();
      if (percentage <= _ctx.small_color_blacklist) {
        _color_blacklist[c] = 1;
      } else {
        break;
      }
    }
  };

  KASSERT(_color_sizes.front() == 0u);
  KASSERT(_color_sizes.back() == graph.n());

  TIMED_SCOPE("Allocation") {
    _matching.clear();
    _matching.resize(graph.total_n());
    tbb::parallel_for<NodeID>(0, graph.total_n(), [&](const NodeID u) {
      _matching[u] = kInvalidGlobalNodeID;
    });
  };
}

HEMClusterer::ClusterArray &
HEMClusterer::cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) {
  KASSERT(_graph == &graph, "must call initialize() before cluster()", assert::always);
  SCOPED_TIMER("Compute HEM clustering");

  for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
    compute_local_matching(c, max_cluster_weight);
    resolve_global_conflicts(c);
  }

  // Unmatched nodes become singleton clusters
  _graph->pfor_all_nodes([&](const NodeID u) {
    if (_matching[u] == kInvalidGlobalNodeID) {
      _matching[u] = _graph->local_to_global_node(u);
    }
  });

  // Validate our matching
  KASSERT(validate_matching(), "matching in inconsistent state", assert::always);

  return _matching;
}

bool HEMClusterer::validate_matching() {
  for (const NodeID u : _graph->nodes()) {
    const GlobalNodeID u_partner = _matching[u];

    KASSERT(_graph->contains_global_node(u_partner), "invalid matching partner for node " << u);
    if (_graph->is_owned_global_node(u_partner)) {
      const NodeID local_partner = _graph->global_to_local_node(u_partner);
      const GlobalNodeID u_global = _graph->local_to_global_node(u);
      KASSERT(
          u == local_partner || _matching[local_partner] == u_partner,
          "invalid clustering structure for node "
              << u << " (global " << u_global << ") matched to node " << local_partner
              << ", which is matched to global node " << _matching[local_partner]
      );
    }
  }

  // Check matched edges between PEs
  struct MatchedEdge {
    GlobalNodeID u;
    GlobalNodeID v;
  };
  mpi::graph::sparse_alltoall_interface_to_ghost<MatchedEdge>(
      *_graph,
      [&](const NodeID u, EdgeID, const NodeID v) -> bool {
        return _matching[u] == _graph->local_to_global_node(v);
      },
      [&](const NodeID u, EdgeID, NodeID) -> MatchedEdge {
        return {_graph->local_to_global_node(u), _matching[u]};
      },
      [&](const auto &r, const PEID pe) {
        for (const auto &[u, v] : r) {
          KASSERT(_graph->contains_global_node(u));
          KASSERT(
              _graph->is_owned_global_node(v), "PE " << pe << " thinks that this PE owns " << v
          );
          const NodeID local_u = _graph->global_to_local_node(u);
          const NodeID local_v = _graph->global_to_local_node(v);

          KASSERT(
              _matching[local_v] == v,
              "invalid clustering structure for edge "
                  << u << " <-> " << v << " (local " << local_u << " <-> " << local_v
                  << "): expected " << v << " to be the leader, but " << v << " is in cluster "
                  << _matching[local_v]
          );
        }
      }
  );

  return true;
}

void HEMClusterer::compute_local_matching(
    const ColorID c, const GlobalNodeWeight max_cluster_weight
) {
  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];
  _graph->pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
    const NodeID u = _color_sorted_nodes[seq_u];
    if (_matching[u] != kInvalidGlobalNodeID) {
      return; // Node already matched
    }

    const NodeWeight u_weight = _graph->node_weight(u);

    // @todo if matching fails due to a race condition, we could try again

    NodeID best_neighbor = 0;
    EdgeWeight best_weight = 0;
    for (const auto [e, v] : _graph->neighbors(u)) {
      // v already matched?
      if (_matching[v] != kInvalidGlobalNodeID) {
        continue;
      }

      // v too heavy?
      const NodeWeight v_weight = _graph->node_weight(v);
      if (u_weight + v_weight > max_cluster_weight && !_ctx.ignore_weight_limit) {
        continue;
      }

      // Already found a better neighbor?
      const EdgeWeight e_weight = _graph->edge_weight(e);
      if (e_weight < best_weight) {
        continue;
      }

      // Match with v
      best_weight = e_weight;
      best_neighbor = v;
    }

    // If we found a good neighbor, try to match with it
    if (best_weight > 0) {
      const GlobalNodeID neighbor_global = _graph->local_to_global_node(best_neighbor);
      GlobalNodeID unmatched = kInvalidGlobalNodeID;
      if (__atomic_compare_exchange_n(
              &_matching[best_neighbor],
              &unmatched,
              neighbor_global,
              true,
              __ATOMIC_SEQ_CST,
              __ATOMIC_SEQ_CST
          )) {
        // @todo if we merge small colors, also use CAS to match our own node
        // and revert matching of best_neighbor if our CAS failed
        __atomic_store_n(&_matching[u], neighbor_global, __ATOMIC_RELAXED);
      }
    }
  });
}

void HEMClusterer::resolve_global_conflicts(const ColorID c) {
  struct MatchRequest {
    NodeID mine;
    NodeID theirs;
    EdgeWeight weight;
  };

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  // @todo avoid O(m), use same "trick" as below?
  auto all_requests = mpi::graph::sparse_alltoall_interface_to_ghost_custom_range_get<MatchRequest>(
      *_graph,
      seq_from,
      seq_to,
      [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },
      [&](const NodeID u, EdgeID, const NodeID v) {
        return _matching[u] == _graph->local_to_global_node(v);
      },
      [&](const NodeID u, const EdgeID e, const NodeID v, const PEID pe) -> MatchRequest {
        const GlobalNodeID v_global = _graph->local_to_global_node(v);
        const NodeID their_v = static_cast<NodeID>(v_global - _graph->offset_n(pe));
        return {u, their_v, _graph->edge_weight(e)};
      }
  );

  parallel::chunked_for(all_requests, [&](MatchRequest &req, PEID) {
    std::swap(req.theirs, req.mine); // Swap roles of theirs and mine

    if (_matching[req.mine] != kInvalidGlobalNodeID) {
      req.mine = kInvalidNodeID; // Reject: local node matched to node
    }
  });

  parallel::chunked_for(all_requests, [&](MatchRequest &req, const PEID pe) {
    if (req.mine == kInvalidNodeID) {
      return;
    }

    KASSERT(_graph->contains_global_node(req.theirs + _graph->offset_n(pe)));
    req.theirs = _graph->global_to_local_node(req.theirs + _graph->offset_n(pe));
    KASSERT(_graph->is_ghost_node(req.theirs));

    GlobalNodeID current_partner = _matching[req.mine];
    GlobalNodeID new_partner = current_partner;
    do {
      const EdgeWeight current_weight = current_partner == kInvalidGlobalNodeID
                                            ? 0
                                            : static_cast<EdgeWeight>(current_partner >> 32);
      if (req.weight <= current_weight) {
        break;
      }
      new_partner = (static_cast<GlobalNodeID>(req.weight) << 32) | req.theirs;
    } while (__atomic_compare_exchange_n(
        &_matching[req.mine],
        &current_partner,
        new_partner,
        true,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST
    ));
  });

  // Create response messages
  parallel::chunked_for(all_requests, [&](MatchRequest &req, const PEID pe) {
    if (req.mine == kInvalidNodeID) {
      return;
    }

    const NodeID winner = _matching[req.mine] & 0xFFFF'FFFF;
    if (req.theirs != winner) {
      // Indicate that the matching failed
      req.mine = kInvalidNodeID;
    }

    req.theirs =
        static_cast<NodeID>(_graph->local_to_global_node(req.theirs) - _graph->offset_n(pe));
  });

  // Normalize our _matching array
  parallel::chunked_for(all_requests, [&](const MatchRequest &req) {
    if (req.mine != kInvalidNodeID) { // Due to the previous step, this should
                                      // only happen once per node
      _matching[req.mine] =
          _graph->local_to_global_node(req.mine); // We become the leader of this cluster
    }
  });

  // Exchange response messages
  auto all_responses = mpi::sparse_alltoall_get<MatchRequest>(all_requests, _graph->communicator());

  parallel::chunked_for(all_responses, [&](MatchRequest &rsp) {
    std::swap(rsp.mine, rsp.theirs); // Swap roles of theirs and mine

    if (rsp.theirs == kInvalidNodeID) {
      // We have to unmatch the ghost node
      _matching[rsp.mine] = kInvalidGlobalNodeID;
    }
  });

  // Synchronize matching:
  // - nodes that where active during this round
  // - their matching partners
  // - interface nodes that got matched by nodes on other PEs
  struct MatchedMessage {
    NodeID node;
    GlobalNodeID partner;
  };

  const PEID size = mpi::get_comm_size(_graph->communicator());
  std::vector<std::vector<MatchedMessage>> sync_msgs(size);
  Marker<> marked(size);

  auto add_node = [&](const NodeID u) {
    marked.reset();
    for (const auto &[e, v] : _graph->neighbors(u)) {
      if (!_graph->is_ghost_node(v)) {
        continue;
      }

      const PEID owner = _graph->ghost_owner(v);
      if (!marked.get(owner)) {
        sync_msgs[owner].push_back({u, _matching[u]});
        marked.set(owner);
      }
    }
  };

  for (const NodeID seq_u : _graph->nodes(seq_from, seq_to)) {
    const NodeID u = _color_sorted_nodes[seq_u];
    const GlobalNodeID partner = _matching[u];
    if (partner != kInvalidGlobalNodeID) {
      add_node(u);

      if (_graph->is_owned_global_node(partner)) {
        const NodeID local_partner = _graph->global_to_local_node(partner);
        if (u != local_partner) {
          add_node(local_partner);
        }
      }
    }
  }

  for (const auto &pe_requests : all_requests) {
    for (const auto &req : pe_requests) {
      if (req.mine != kInvalidNodeID) {
        add_node(req.mine);
      }
    }
  }

  mpi::sparse_alltoall<MatchedMessage>(
      sync_msgs,
      [&](const auto &r, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, r.size(), [&](const std::size_t i) {
          const auto [local_node_on_pe, partner] = r[i];
          const auto global_node =
              static_cast<GlobalNodeID>(_graph->offset_n(pe) + local_node_on_pe);
          const NodeID local_node = _graph->global_to_local_node(global_node);
          _matching[local_node] = partner;
        });
      },
      _graph->communicator()
  );
}
} // namespace kaminpar::dist
