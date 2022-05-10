#pragma once

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <kassert/kassert.hpp>
#include <omp.h>
#include <tbb/global_control.h>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "mpi_graph.h"

#define SINGLE_THREADED_TEST                            \
    omp_set_num_threads(1);                             \
    auto GC = tbb::global_control {                     \
        tbb::global_control::max_allowed_parallelism, 1 \
    }

namespace dkaminpar::test {
class DistributedGraphFixture : public ::testing::Test {
protected:
    void SetUp() override {
        std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD);
    }

    GlobalNodeID next(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
        return (u + step) % n;
    }

    GlobalNodeID prev(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
        return (u < step) ? n + u - step : u - step;
    }

    PEID size;
    PEID rank;
};

/*
 * Utility function for graphs
 */
namespace graph {
//! Create a partitioned graph based on block labels of owned nodes
inline DistributedPartitionedGraph
make_partitioned_graph(const DistributedGraph& graph, const BlockID k, const std::vector<BlockID>& local_partition) {
    scalable_vector<parallel::Atomic<BlockID>> partition(graph.total_n());
    scalable_vector<BlockWeight>               local_block_weights(k);

    std::copy(local_partition.begin(), local_partition.end(), partition.begin());
    for (const NodeID u: graph.nodes()) {
        local_block_weights[partition[u]] += graph.node_weight(u);
    }

    scalable_vector<BlockWeight> global_block_weights_nonatomic(k);
    mpi::allreduce(local_block_weights.data(), global_block_weights_nonatomic.data(), k, MPI_SUM, MPI_COMM_WORLD);

    scalable_vector<parallel::Atomic<BlockWeight>> block_weights(k);
    std::copy(global_block_weights_nonatomic.begin(), global_block_weights_nonatomic.end(), block_weights.begin());

    struct NodeBlock {
        GlobalNodeID global_node;
        BlockID      block_weights;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<NodeBlock>(
        graph,
        [&](const NodeID u) {
            return NodeBlock{graph.local_to_global_node(u), local_partition[u]};
        },
        [&](const auto& buffer) {
            for (const auto& [global_node, block]: buffer) {
                partition[graph.global_to_local_node(global_node)] = block;
            }
        });

    return {&graph, k, std::move(partition), std::move(block_weights)};
}

//! Return the id of the edge connecting two adjacent nodes \c u and \c v in \c graph, found by linear search.
inline std::pair<EdgeID, EdgeID> get_edge_by_endpoints(const DistributedGraph& graph, const NodeID u, const NodeID v) {
    EdgeID forward_edge  = kInvalidEdgeID;
    EdgeID backward_edge = kInvalidEdgeID;

    if (graph.is_owned_node(u)) {
        for (const auto [cur_e, cur_v]: graph.neighbors(u)) {
            if (cur_v == v) {
                forward_edge = cur_e;
                break;
            }
        }
    }

    if (graph.is_owned_node(v)) {
        for (const auto [cur_e, cur_u]: graph.neighbors(v)) {
            if (cur_u == u) {
                backward_edge = cur_e;
                break;
            }
        }
    }

    // one of those edges might now exist due to ghost nodes
    return {forward_edge, backward_edge};
}

//! Return the id of the edge connecting two adjacent nodes \c u and \c v given by their global id in \c graph,
//! found by linear search
inline std::pair<EdgeID, EdgeID>
get_edge_by_endpoints_global(const DistributedGraph& graph, const GlobalNodeID u, const GlobalNodeID v) {
    return get_edge_by_endpoints(graph, graph.global_to_local_node(u), graph.global_to_local_node(v));
}

//! Based on some graph, build a new graph with modified edge weights.
inline DistributedGraph
change_edge_weights(DistributedGraph graph, const std::vector<std::pair<EdgeID, EdgeWeight>>& changes) {
    auto edge_weights = graph.take_edge_weights();
    if (edge_weights.empty()) {
        edge_weights.resize(graph.m(), 1);
    }

    for (const auto& [e, weight]: changes) {
        if (e != kInvalidEdgeID) {
            edge_weights[e] = weight;
        }
    }

    return {
        graph.take_node_distribution(),
        graph.take_edge_distribution(),
        graph.take_nodes(),
        graph.take_edges(),
        graph.take_node_weights(),
        std::move(edge_weights),
        graph.take_ghost_owner(),
        graph.take_ghost_to_global(),
        graph.take_global_to_ghost(),
        false,
        graph.communicator()};
}

inline DistributedGraph change_edge_weights_by_endpoints(
    DistributedGraph graph, const std::vector<std::tuple<NodeID, NodeID, EdgeWeight>>& changes) {
    std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
    for (const auto& [u, v, weight]: changes) {
        const auto [forward_edge, backward_edge] = get_edge_by_endpoints(graph, u, v);
        edge_id_changes.emplace_back(forward_edge, weight);
        edge_id_changes.emplace_back(backward_edge, weight);
    }

    return change_edge_weights(std::move(graph), edge_id_changes);
}

inline DistributedGraph change_edge_weights_by_global_endpoints(
    DistributedGraph graph, const std::vector<std::tuple<GlobalNodeID, GlobalNodeID, EdgeWeight>>& changes) {
    SET_DEBUG(true);
    std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
    for (const auto& [u, v, weight]: changes) {
        const auto real_u                        = u % graph.global_n();
        const auto real_v                        = v % graph.global_n();
        const auto [forward_edge, backward_edge] = get_edge_by_endpoints_global(graph, real_u, real_v);
        DBG << u << "/" << real_u << " / " << graph.global_to_local_node(real_u) << " -- " << v << " / " << real_v
            << " / " << graph.global_to_local_node(real_v) << " == " << weight << " ----- " << forward_edge << " <-> "
            << backward_edge;
        edge_id_changes.emplace_back(forward_edge, weight);
        edge_id_changes.emplace_back(backward_edge, weight);
    }

    return change_edge_weights(std::move(graph), edge_id_changes);
}

//! Based on some graph, build a new graph with modified node weights.
inline DistributedGraph
change_node_weights(DistributedGraph graph, const std::vector<std::pair<NodeID, NodeWeight>>& changes) {
    auto node_weights = graph.take_node_weights();
    if (node_weights.empty()) {
        node_weights.resize(graph.total_n(), 1);
    }

    for (const auto& [u, weight]: changes) {
        KASSERT(u < node_weights.size(), "", assert::always);
        node_weights[u] = weight;
    }

    return {
        graph.take_node_distribution(),
        graph.take_edge_distribution(),
        graph.take_nodes(),
        graph.take_edges(),
        std::move(node_weights),
        graph.take_edge_weights(),
        graph.take_ghost_owner(),
        graph.take_ghost_to_global(),
        graph.take_global_to_ghost(),
        false,
        graph.communicator()};
}

/**
 * Given an array \c info with one element per global node, build a local array on each PE containing copies of the
 * information associated with each node and ghost node.
 *
 * @tparam T Information associated with nodes.
 * @tparam Buffer Vector type.
 * @param graph Distributed graph.
 * @param info Information associated with nodes.
 * @return A local array of size \c total_n containing the information associated with each owned and ghost node on
 * the PE.
 */
template <typename Container>
Container distribute_node_info(const DistributedGraph& graph, const Container& global_info) {
    KASSERT(global_info.size() == graph.global_n());

    Container local_info(graph.total_n());
    for (const NodeID u: graph.all_nodes()) {
        local_info[u] = global_info[graph.local_to_global_node(u)];
    }
    return local_info;
}

/**
 * Changes nodes weights such that node \c u has weight \c 2^u. Thus, the weight of each node is unique and the sum
 * of node weights can be decomposed.
 * @param graph Graph for which the node weights are changed, cannot have more than 31 nodes (globally) since node
 * weights are usually 32 bit signed integers.
 * @return Graph with unique node weights.
 */
inline DistributedGraph use_pow_global_id_as_node_weights(DistributedGraph graph) {
    KASSERT(graph.global_n() <= 31u, "graph is too large: can have at most 30 nodes", assert::always);

    scalable_vector<NodeWeight> new_node_weights(graph.total_n());
    for (const NodeID u: graph.all_nodes()) {
        new_node_weights[u] = 1 << graph.local_to_global_node(u);
    }

    return {
        graph.take_node_distribution(),
        graph.take_edge_distribution(),
        graph.take_nodes(),
        graph.take_edges(),
        std::move(new_node_weights),
        graph.take_edge_weights(),
        graph.take_ghost_owner(),
        graph.take_ghost_to_global(),
        graph.take_global_to_ghost(),
        false,
        graph.communicator()};
}

inline DistributedGraph use_global_id_as_node_weight(DistributedGraph graph) {
    scalable_vector<NodeWeight> new_node_weights(graph.total_n());
    for (const NodeID u: graph.all_nodes()) {
        new_node_weights[u] = graph.local_to_global_node(u) + 1;
    }

    return {
        graph.take_node_distribution(),
        graph.take_edge_distribution(),
        graph.take_nodes(),
        graph.take_edges(),
        std::move(new_node_weights),
        graph.take_edge_weights(),
        graph.take_ghost_owner(),
        graph.take_ghost_to_global(),
        graph.take_global_to_ghost(),
        false,
        graph.communicator()};
}

struct NodeWeightIdentifiedEdge {
    NodeWeightIdentifiedEdge(const NodeWeight u_weight, const EdgeWeight weight, const NodeWeight v_weight)
        : u_weight(u_weight),
          weight(weight),
          v_weight(v_weight) {}

    NodeWeight u_weight;
    EdgeWeight weight;
    NodeWeight v_weight;
};

namespace internal {
inline std::vector<NodeWeightIdentifiedEdge> graph_to_edge_list(const DistributedGraph& graph) {
    const auto shm_graph = dkaminpar::graph::allgather(graph);

    std::vector<NodeWeightIdentifiedEdge> list;
    for (const NodeID u: shm_graph.nodes()) {
        for (const auto [e, v]: shm_graph.neighbors(u)) {
            if (u < v) {
                list.emplace_back(shm_graph.node_weight(u), shm_graph.edge_weight(e), shm_graph.node_weight(v));
            }
        }
    }
    return list;
}

/**
 * Adds reverse edges to a list of undirected edges.
 * @param list List with edges only in one direction.
 * @return List containing edges in both directions.
 */
inline std::vector<NodeWeightIdentifiedEdge> make_edge_list_undirected(std::vector<NodeWeightIdentifiedEdge> list) {
    const std::size_t initial_size = list.size();
    for (std::size_t i = 0; i < initial_size; ++i) {
        list.emplace_back(list[i].v_weight, list[i].weight, list[i].u_weight);
    }
    return list;
}

/**
 * Checks whether the given distributed graph with node weight identifiers is isomorphic to the specified edge list.
 * The edge list describes edges by their weight and by the (unique) node weights of their endpoints.
 * @param d_lhs Distributed graph to check.
 * @param rhs Global edge list.
 * @return Pair of two list: first contains edges in \c d_lhs not in \c rhs, second contains edges in
 * \c rhs not in \c d_lhs.
 */
inline std::pair<std::vector<NodeWeightIdentifiedEdge>, std::vector<NodeWeightIdentifiedEdge>>
check_isomorphic(const DistributedGraph& d_lhs, const std::vector<NodeWeightIdentifiedEdge>& rhs) {
    // allgather graph
    auto lhs = dkaminpar::graph::allgather(d_lhs);

    std::vector<bool> found_from_lhs(lhs.m());
    std::vector<bool> found_from_rhs(rhs.size());

    // search in lhs
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        const auto [u_weight, e_weight, v_weight] = rhs[i];

        for (const NodeID u: lhs.nodes()) {
            if (lhs.node_weight(u) != u_weight) {
                continue;
            }
            for (const auto [e, v]: lhs.neighbors(u)) {
                if (lhs.edge_weight(e) != e_weight) {
                    continue;
                }
                if (lhs.node_weight(v) != v_weight) {
                    continue;
                }
                found_from_rhs[i] = true;
                break;
            }
        }
    }

    // search in rhs
    for (const NodeID u: lhs.nodes()) {
        for (const auto [e, v]: lhs.neighbors(u)) {
            const NodeWeight u_weight = lhs.node_weight(u);
            const EdgeWeight e_weight = lhs.edge_weight(e);
            const NodeWeight v_weight = lhs.node_weight(v);

            for (std::size_t i = 0; i < rhs.size(); ++i) {
                if (u_weight != rhs[i].u_weight) {
                    continue;
                }
                if (e_weight != rhs[i].weight) {
                    continue;
                }
                if (v_weight != rhs[i].v_weight) {
                    continue;
                }
                found_from_lhs[e] = true;
                break;
            }
        }
    }

    // build diff pair
    std::pair<std::vector<NodeWeightIdentifiedEdge>, std::vector<NodeWeightIdentifiedEdge>> diff;

    for (const NodeID u: lhs.nodes()) {
        for (const auto [e, v]: lhs.neighbors(u)) {
            if (!found_from_lhs[e]) {
                diff.first.emplace_back(lhs.node_weight(u), lhs.edge_weight(e), lhs.node_weight(v));
            }
        }
    }

    for (std::size_t i = 0; i < rhs.size(); ++i) {
        if (!found_from_rhs[i]) {
            diff.second.push_back(rhs[i]);
        }
    }

    return diff;
}
} // namespace internal

inline void
expect_isomorphic(const DistributedGraph& lhs, const std::vector<NodeWeightIdentifiedEdge>& undirected_rhs) {
    const auto rhs = internal::make_edge_list_undirected(undirected_rhs);

    EXPECT_EQ(lhs.global_m(), rhs.size());

    const auto [lhs_diff, rhs_diff] = internal::check_isomorphic(lhs, rhs);
    if (!lhs_diff.empty() || !rhs_diff.empty()) {
        std::ostringstream oss;

        oss << "Expected graphs to be isomorphic, but they are not:\n"
            << "Edges contained in LHS that do not exist in RHS:\n";
        for (const auto [u_weight, e_weight, v_weight]: lhs_diff) {
            oss << "- " << u_weight << " --> " << v_weight << " with weight " << e_weight << "\n";
        }
        oss << "Edges contained in RHS that do not exist in LHS:\n";
        for (const auto [u_weight, e_weight, v_weight]: rhs_diff) {
            oss << "- " << u_weight << " --> " << v_weight << " with weight " << e_weight << "\n";
        }

        LOG << oss.str();
    }

    EXPECT_TRUE(lhs_diff.empty());
    EXPECT_TRUE(rhs_diff.empty());
}

inline void expect_empty(const DistributedGraph& lhs) {
    expect_isomorphic(lhs, std::vector<NodeWeightIdentifiedEdge>{});
}

inline void expect_isomorphic(const DistributedGraph& lhs, const DistributedGraph& rhs) {
    const auto rhs_edge_list = internal::graph_to_edge_list(rhs);
    expect_isomorphic(lhs, rhs_edge_list);
}
} // namespace graph

namespace fixtures3PE {
class DistributedNullGraph : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0    = 0;
        graph = dkaminpar::graph::Builder{MPI_COMM_WORLD}.initialize({0, 0, 0, 0}).finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

class DistributedGraphWith9NodesAnd0Edges : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0    = 3 * rank;
        graph = dkaminpar::graph::Builder{MPI_COMM_WORLD}
                    .initialize({0, 3, 6, 9})
                    .create_node(1)
                    .create_node(1)
                    .create_node(1)
                    .finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

class DistributedGraphWith900NodesAnd0Edges : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0 = 300 * rank;
        dkaminpar::graph::Builder builder{MPI_COMM_WORLD};
        builder.initialize({0, 300, 600, 900});
        for (NodeID u = 0; u < 300; ++u) {
            builder.create_node(1);
        }
        graph = builder.finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

//  0---1-#-3---4
//  |\ /  #  \ /|
//  | 2---#---5 |
//  |  \  #  /  |
// ###############
//  |    \ /    |
//  |     8     |
//  |    / \    |
//  +---7---6---+
class DistributedTriangles : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0    = 3 * rank;
        graph = dkaminpar::graph::Builder{MPI_COMM_WORLD}
                    .initialize({0, 3, 6, 9})
                    .create_node(1)
                    .create_edge(1, n0 + 1)
                    .create_edge(1, n0 + 2)
                    .create_edge(1, prev(n0, 2, 9))
                    .create_node(1)
                    .create_edge(1, n0)
                    .create_edge(1, n0 + 2)
                    .create_edge(1, next(n0 + 1, 2, 9))
                    .create_node(1)
                    .create_edge(1, n0)
                    .create_edge(1, n0 + 1)
                    .create_edge(1, next(n0 + 2, 3, 9))
                    .create_edge(1, prev(n0 + 2, 3, 9))
                    .finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

// 0-#-1-#-2
class DistributedPathOneNodePerPE : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0           = rank;
        auto builder = dkaminpar::graph::Builder{MPI_COMM_WORLD}.initialize({0, 1, 2, 3}).create_node(1);

        if (rank == 0) {
            builder.create_edge(1, 1);
        } else if (rank == 1) {
            builder.create_edge(1, 0);
            builder.create_edge(1, 2);
        } else {
            builder.create_edge(1, 1);
        }

        graph = builder.finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

// 0--1-#-2--3-#-4--5
class DistributedPathTwoNodesPerPE : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();
        KASSERT(size == 3, "must be tested on three PEs", assert::always);

        n0 = 2 * rank;
        dkaminpar::graph::Builder builder{MPI_COMM_WORLD};
        builder.initialize({0, 2, 4, 6});
        builder.create_node(1);
        if (rank > 0) {
            builder.create_edge(1, prev(n0, 1, 6));
        }
        builder.create_edge(1, n0 + 1);
        builder.create_node(1);
        builder.create_edge(1, n0);
        if (rank + 1 < size) {
            builder.create_edge(1, next(n0 + 1, 1, 6));
        }
        graph = builder.finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};

// +-------------+
// +------+      |
// 0--1 # 2--3 # 4--5
//        +------+
class UnsortedDistributedPath : public DistributedGraphFixture {
protected:
    void SetUp() override {
        DistributedGraphFixture::SetUp();

        n0 = 2 * rank;

        dkaminpar::graph::Builder builder{MPI_COMM_WORLD};
        builder.initialize(2);

        builder.create_node(1);
        switch (rank) {
            case 0:
                builder.create_edge(1, 2);
                builder.create_edge(1, 4);
                break;

            case 1:
                builder.create_edge(1, 0);
                builder.create_edge(1, 4);
                break;

            case 2:
                builder.create_edge(1, 0);
                builder.create_edge(1, 2);
                break;
        }
        builder.create_edge(1, n0 + 1);

        builder.create_node(1);
        builder.create_edge(1, n0);

        graph = builder.finalize();
    }

    DistributedGraph graph;
    GlobalNodeID     n0;
};
} // namespace fixtures3PE
} // namespace dkaminpar::test
