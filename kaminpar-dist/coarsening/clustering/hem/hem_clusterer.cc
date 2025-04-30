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

//
// Implementation
//

struct HEMClustererMemoryContext {
  NoinitVector<std::uint8_t> color_blacklist;
  NoinitVector<ColorID> color_sizes;
  NoinitVector<NodeID> color_sorted_nodes;
};

template <typename Graph> class HEMClustererImpl {
public:
  explicit HEMClustererImpl(const Context &ctx) : _input_ctx(ctx), _ctx(ctx.coarsening.hem) {}

  void setup(HEMClustererMemoryContext &memory_context) {
    _color_blacklist = std::move(memory_context.color_blacklist);
    _color_sizes = std::move(memory_context.color_sizes);
    _color_sorted_nodes = std::move(memory_context.color_sorted_nodes);
  }

  HEMClustererMemoryContext release() {
    return {
        std::move(_color_blacklist),
        std::move(_color_sizes),
        std::move(_color_sorted_nodes),
    };
  }

  void set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void cluster(StaticArray<GlobalNodeID> &matching, const Graph &graph) {
    _matching = std::move(matching);
    _graph = &graph;

    initialize_coloring();

    SCOPED_TIMER("Compute HEM clustering");

    tbb::parallel_for<NodeID>(0, graph.total_n(), [&](const NodeID u) {
      matching[u] = kInvalidGlobalNodeID;
    });

    for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
      compute_local_matching(c, _max_cluster_weight);
      resolve_global_conflicts(c);
    }

    _graph->pfor_all_nodes([&](const NodeID u) {
      if (matching[u] == kInvalidGlobalNodeID) {
        matching[u] = _graph->local_to_global_node(u);
      }
    });

    KASSERT(validate_matching(), "matching in inconsistent state", assert::always);

    matching = std::move(_matching);
  }

private:
  void initialize_coloring() {
    SCOPED_TIMER("Initialize HEM clustering");

    const auto coloring = [&] {
      // Graph is already sorted by a coloring -> reconstruct this coloring
      // @todo if we always want to do this, optimize this refiner
      if (_graph->color_sorted()) {
        LOG << "Graph sorted by colors: using precomputed coloring";

        // We do not actually need the colors for ghost nodes
        NoinitVector<ColorID> coloring(_graph->n());

        // @todo parallelize
        NodeID pos = 0;
        for (ColorID c = 0; c < _graph->number_of_colors(); ++c) {
          const std::size_t size = _graph->color_size(c);
          std::fill(coloring.begin() + pos, coloring.begin() + pos + size, c);
          pos += size;
        }

        return coloring;
      }

      // Otherwise, compute a coloring now
      LOG << "Computing new coloring";
      return compute_node_coloring_sequentially(*_graph, _ctx.chunks.compute(_input_ctx.parallel));
    }();

    const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    const ColorID num_colors = mpi::allreduce(num_local_colors, MPI_MAX, _graph->communicator());

    TIMED_SCOPE("Allocation") {
      _color_sorted_nodes.resize(_graph->n());
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
      if (_graph->color_sorted()) {
        const auto &color_sizes = _graph->get_color_sizes();
        _color_sizes.assign(color_sizes.begin(), color_sizes.end());
      } else {
        _graph->pfor_nodes([&](const NodeID u) {
          const ColorID c = coloring[u];
          KASSERT(c < num_colors);
          __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
        });
        parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
      }
    };

    TIMED_SCOPE("Sort nodes") {
      if (_graph->color_sorted()) {
        // @todo parallelize
        std::iota(_color_sorted_nodes.begin(), _color_sorted_nodes.end(), 0);
      } else {
        _graph->pfor_nodes([&](const NodeID u) {
          const ColorID c = coloring[u];
          const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
          KASSERT(i < _color_sorted_nodes.size());
          _color_sorted_nodes[i] = u;
        });
      }
    };

    TIMED_SCOPE("Compute color blacklist") {
      if (_ctx.small_color_blacklist == 0 ||
          (_ctx.only_blacklist_input_level && _graph->global_n() != _input_ctx.partition.global_n
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
          _graph->communicator()
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
        const double percentage = 1.0 * excluded_so_far / _graph->global_n();
        if (percentage <= _ctx.small_color_blacklist) {
          _color_blacklist[c] = 1;
        } else {
          break;
        }
      }
    };

    KASSERT(_color_sizes.front() == 0u);
    KASSERT(_color_sizes.back() == _graph->n());
  }

  void compute_local_matching(ColorID c, GlobalNodeWeight max_cluster_weight) {
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
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight e_weight) {
        // v already matched?
        if (_matching[v] != kInvalidGlobalNodeID) {
          return;
        }

        // v too heavy?
        const NodeWeight v_weight = _graph->node_weight(v);
        if (u_weight + v_weight > max_cluster_weight && !_ctx.ignore_weight_limit) {
          return;
        }

        // Already found a better neighbor?
        if (e_weight < best_weight) {
          return;
        }

        // Match with v
        best_weight = e_weight;
        best_neighbor = v;
      });

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

  void resolve_global_conflicts(ColorID c) {
    struct MatchRequest {
      NodeID mine;
      NodeID theirs;
      EdgeWeight weight;
    };

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to = _color_sizes[c + 1];

    // @todo avoid O(m), use same "trick" as below?
    auto all_requests =
        mpi::graph::sparse_alltoall_interface_to_ghost_custom_range_get<MatchRequest>(
            *_graph,
            seq_from,
            seq_to,
            [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },
            [&](const NodeID u, const NodeID v, EdgeWeight) {
              return _matching[u] == _graph->local_to_global_node(v);
            },
            [&](const NodeID u, const NodeID v, const EdgeWeight w, const PEID pe) {
              const GlobalNodeID v_global = _graph->local_to_global_node(v);
              const NodeID their_v = static_cast<NodeID>(v_global - _graph->offset_n(pe));
              return MatchRequest(u, their_v, w);
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
    auto all_responses =
        mpi::sparse_alltoall_get<MatchRequest>(all_requests, _graph->communicator());

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
      _graph->adjacent_nodes(u, [&](const NodeID v) {
        if (!_graph->is_ghost_node(v)) {
          return;
        }

        const PEID owner = _graph->ghost_owner(v);
        if (!marked.get(owner)) {
          sync_msgs[owner].push_back({u, _matching[u]});
          marked.set(owner);
        }
      });
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

  bool validate_matching() {
    for (const NodeID u : _graph->nodes()) {
      const GlobalNodeID u_partner = _matching[u];

      KASSERT(_graph->contains_global_node(u_partner), "invalid matching partner for node " << u);
      if (_graph->is_owned_global_node(u_partner)) {
        [[maybe_unused]] const NodeID local_partner = _graph->global_to_local_node(u_partner);
        [[maybe_unused]] const GlobalNodeID u_global = _graph->local_to_global_node(u);
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
        [&](const NodeID u, const NodeID v, EdgeWeight) -> bool {
          return _matching[u] == _graph->local_to_global_node(v);
        },
        [&](const NodeID u, NodeID, EdgeWeight) -> MatchedEdge {
          return {_graph->local_to_global_node(u), _matching[u]};
        },
        [&](const auto &r, [[maybe_unused]] const PEID pe) {
          for (const auto &[u, v] : r) {
            KASSERT(_graph->contains_global_node(u));
            KASSERT(
                _graph->is_owned_global_node(v), "PE " << pe << " thinks that this PE owns " << v
            );
            [[maybe_unused]] const NodeID local_u = _graph->global_to_local_node(u);
            [[maybe_unused]] const NodeID local_v = _graph->global_to_local_node(v);

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

  const Context &_input_ctx;
  const HEMCoarseningContext &_ctx;

  const Graph *_graph;

  NoinitVector<std::uint8_t> _color_blacklist;
  NoinitVector<ColorID> _color_sizes;
  NoinitVector<NodeID> _color_sorted_nodes;

  GlobalNodeWeight _max_cluster_weight = 0;
  StaticArray<GlobalNodeID> _matching;
};

//
// Private interface
//

class HEMClustererImplWrapper {
public:
  HEMClustererImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<HEMClustererImpl<DistributedCSRGraph>>(ctx)),
        _compressed_impl(std::make_unique<HEMClustererImpl<DistributedCompressedGraph>>(ctx)) {}

  void set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) {
    _csr_impl->set_max_cluster_weight(max_cluster_weight);
    _compressed_impl->set_max_cluster_weight(max_cluster_weight);
  }

  void cluster(StaticArray<GlobalNodeID> &matching, const DistributedGraph &graph) {
    const auto compute_cluster = [&](auto &impl, const auto &graph) {
      impl.setup(_memory_context);
      impl.cluster(matching, graph);
      _memory_context = impl.release();
    };

    graph.reified(
        [&](const DistributedCSRGraph &csr_graph) {
          HEMClustererImpl<DistributedCSRGraph> &impl = *_csr_impl;
          compute_cluster(impl, csr_graph);
        },
        [&](const DistributedCompressedGraph &compressed_graph) {
          HEMClustererImpl<DistributedCompressedGraph> &impl = *_compressed_impl;
          compute_cluster(impl, compressed_graph);
        }
    );
  }

private:
  HEMClustererMemoryContext _memory_context;
  std::unique_ptr<HEMClustererImpl<DistributedCSRGraph>> _csr_impl;
  std::unique_ptr<HEMClustererImpl<DistributedCompressedGraph>> _compressed_impl;
};

//
// Public interface
//

HEMClusterer::HEMClusterer(const Context &ctx)
    : _impl_wrapper(std::make_unique<HEMClustererImplWrapper>(ctx)) {}

HEMClusterer::~HEMClusterer() = default;

void HEMClusterer::set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) {
  _impl_wrapper->set_max_cluster_weight(max_cluster_weight);
}

void HEMClusterer::cluster(StaticArray<GlobalNodeID> &matching, const DistributedGraph &graph) {
  _impl_wrapper->cluster(matching, graph);
}

} // namespace kaminpar::dist
