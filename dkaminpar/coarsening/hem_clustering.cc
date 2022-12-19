#include "dkaminpar/coarsening/hem_clustering.h"

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/mpi/graph_communication.h"

#include "common/parallel/loops.h"
#include "common/timer.h"

namespace kaminpar::dist {
HEMClustering::HEMClustering(const Context& ctx) : _input_ctx(ctx), _ctx(ctx.coarsening.hem) {}

void HEMClustering::initialize(const DistributedGraph& graph) {
    mpi::barrier(graph.communicator());

    SCOPED_TIMER("Colored LP refinement");
    SCOPED_TIMER("Initialization");

    const auto coloring = [&] {
        // Graph is already sorted by a coloring -> reconstruct this coloring
        // @todo if we always want to do this, optimize this refiner
        if (graph.color_sorted()) {
            LOG << "Graph sorted by colors: using precomputed coloring";

            NoinitVector<ColorID> coloring(graph.n()); // We do not actually need the colors for ghost nodes

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
        return compute_node_coloring_sequentially(graph, _ctx.num_coloring_chunks);
    }();

    const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    const ColorID num_colors       = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());

    TIMED_SCOPE("Allocation") {
        _color_sorted_nodes.resize(graph.n());
        _color_sizes.resize(num_colors + 1);
        _color_blacklist.resize(num_colors);
        tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
            _color_sorted_nodes[i] = 0;
        });
        tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) { _color_sizes[i] = 0; });
        tbb::parallel_for<std::size_t>(0, _color_blacklist.size(), [&](const std::size_t i) {
            _color_blacklist[i] = 0;
        });
    };

    TIMED_SCOPE("Count color sizes") {
        if (graph.color_sorted()) {
            const auto& color_sizes = graph.get_color_sizes();
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
                const ColorID     c = coloring[u];
                const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
                KASSERT(i < _color_sorted_nodes.size());
                _color_sorted_nodes[i] = u;
            });
        }
    };

    TIMED_SCOPE("Compute color blacklist") {
        if (_ctx.small_color_blacklist == 0
            || (_ctx.only_blacklist_input_level && graph.global_n() != _input_ctx.partition.graph.global_n())) {
            return;
        }

        NoinitVector<GlobalNodeID> global_color_sizes(num_colors);
        tbb::parallel_for<ColorID>(0, num_colors, [&](const ColorID c) {
            global_color_sizes[c] = _color_sizes[c + 1] - _color_sizes[c];
        });
        MPI_Allreduce(
            MPI_IN_PLACE, global_color_sizes.data(), asserting_cast<int>(num_colors), mpi::type::get<GlobalNodeID>(),
            MPI_SUM, graph.communicator()
        );

        // @todo parallelize the rest of this section
        std::vector<ColorID> sorted_by_size(num_colors);
        std::iota(sorted_by_size.begin(), sorted_by_size.end(), 0);
        std::sort(sorted_by_size.begin(), sorted_by_size.end(), [&](const ColorID lhs, const ColorID rhs) {
            return global_color_sizes[lhs] < global_color_sizes[rhs];
        });

        GlobalNodeID excluded_so_far = 0;
        for (const ColorID c: sorted_by_size) {
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
        _matching.resize(graph.total_n(), kInvalidGlobalNodeID);
        _matched.clear();
        _matched.resize(graph.n());
    };
}

const HEMClustering::AtomicClusterArray&
HEMClustering::compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) {
    _graph = &graph;

    initialize(graph);

    for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
        compute_local_matching(c, max_cluster_weight);
        resolve_global_conflicts(c);

        // After the first two steps, _matching[u] holds the global node ID of u's matching partner 
        // The next step replaces the _matching value of the smaller node ID with its own global node ID,
        // turning the _matching array into a cluster array
        turn_into_clustering(c);
    }

    // Unmatched nodes become singleton clusters
    _graph->pfor_all_nodes([&](const NodeID u) {
        if (_matching[u] == kInvalidGlobalNodeID) {
            _matching[u] = _graph->local_to_global_node(u);
        }
    });

    return _matching;
}

void HEMClustering::compute_local_matching(const ColorID c, const GlobalNodeWeight max_cluster_weight) {
    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];
    _graph->pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
        const NodeID u = _color_sorted_nodes[seq_u];
        if (_matching[u] != kInvalidGlobalNodeID) {
            return; // Node already matched
        }

        const NodeWeight u_weight = _graph->node_weight(u);

        // @todo if matching fails due to a race condition, we could try again

        NodeID     best_neighbor = 0;
        EdgeWeight best_weight   = 0;
        for (const auto [e, v]: _graph->neighbors(u)) {
            // v already matched?
            if (_matching[v] != kInvalidGlobalNodeID) {
                continue;
            }

            // v too heavy?
            const NodeWeight v_weight = _graph->node_weight(v);
            if (u_weight + v_weight > max_cluster_weight) {
                continue;
            }

            // Already found a better neighbor?
            const EdgeWeight e_weight = _graph->edge_weight(e);
            if (e_weight < best_weight) {
                continue;
            }

            // Match with v
            best_weight   = e_weight;
            best_neighbor = v;
        }

        // If we found a good neighbor, try to match with it
        if (best_weight > 0) {
            const GlobalNodeID u_global  = _graph->local_to_global_node(u);
            GlobalNodeID       unmatched = kInvalidGlobalNodeID;
            if (!_matching[best_neighbor].compare_exchange_strong(unmatched, u_global)) {
                return; // Matching failed due to a race condition
            }

            // @todo if we merge small colors, also use CAS to match our own node and revert matching of best_neighbor
            // if our CAS failed
            _matching[u] = _graph->local_to_global_node(best_neighbor);
        }
    });
}

void HEMClustering::resolve_global_conflicts(const ColorID c) {
    struct MatchRequest {
        NodeID     mine;
        NodeID     theirs;
        EdgeWeight weight;
    };

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    auto all_requests = mpi::graph::sparse_alltoall_interface_to_ghost_get<MatchRequest>(
        *_graph, seq_from, seq_to,
        [&](const NodeID seq_u) {
            const NodeID u = _color_sorted_nodes[seq_u];
            return _matching[u] != kInvalidGlobalNodeID && !_graph->is_owned_global_node(_matching[u]);
        },
        [&](const NodeID u, const EdgeID e, const NodeID v, const PEID pe) -> MatchRequest {
            const GlobalNodeID v_global = _graph->local_to_global_node(v);
            const NodeID       their_v  = static_cast<NodeID>(v_global - _graph->offset_n(pe));
            return {u, their_v, _graph->edge_weight(e)};
        }
    );

    parallel::chunked_for(all_requests, [&](MatchRequest& req, const PEID pe) {
        std::swap(req.theirs, req.mine);
        req.theirs = _graph->global_to_local_node(req.theirs + _graph->offset_n(pe));

        _matched[req.mine] = 1;

        GlobalNodeID current_partner = _matching[req.mine];
        GlobalNodeID new_partner     = current_partner;
        do {
            const EdgeWeight current_weight =
                current_partner == kInvalidGlobalNodeID ? 0 : static_cast<EdgeWeight>(current_partner >> 32);
            if (req.weight <= current_weight) {
                break;
            }
            new_partner = (static_cast<GlobalNodeID>(req.weight) << 32) | req.theirs;
        } while (_matching[req.mine].compare_exchange_strong(current_partner, new_partner));
    });

    // Create response messages
    parallel::chunked_for(all_requests, [&](MatchRequest& req, const PEID pe) {
        req.theirs = static_cast<NodeID>(_graph->local_to_global_node(req.theirs) - _graph->offset_n(pe));

        const NodeID winner = _matching[req.mine] & 0xFFFF'FFFF;
        if (req.theirs != winner) {
            req.mine = kInvalidNodeID;
        }
    });

    // Normalize our _matching array
    parallel::chunked_for(all_requests, [&](const MatchRequest& req) {
        std::uint8_t one = 1;
        if (_matched[req.mine].compare_exchange_strong(one, 0)) {
            _matching[req.mine] = _graph->local_to_global_node(_matched[req.mine] & 0xFFFF'FFFF);
        }
    });

    // Exchange response messages
    auto all_responses = mpi::sparse_alltoall_get<MatchRequest>(all_requests, _graph->communicator());

    parallel::chunked_for(all_responses, [&](MatchRequest& rsp) {
        std::swap(rsp.mine, rsp.theirs);
        if (rsp.theirs == kInvalidNodeID) {
            // We have to unmatch the ghost node
            _matching[rsp.mine] = kInvalidGlobalNodeID;
        }
    });
}

void HEMClustering::turn_into_clustering(const ColorID c) {
    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];
    _graph->pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
        const NodeID       u        = _color_sorted_nodes[seq_u];
        const GlobalNodeID u_global = _graph->local_to_global_node(u);
        const GlobalNodeID partner  = _matching[u];
        if (partner == kInvalidGlobalNodeID || partner == u_global) {
            return;
        }

        if (!_graph->is_owned_global_node(partner) || u_global < partner) {
            _matching[u] = u_global;
        }
    });
}
} // namespace kaminpar::dist
