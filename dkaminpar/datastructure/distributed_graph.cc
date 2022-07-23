/*******************************************************************************
 * @file:   distributed_graph.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Static distributed graph data structure.
 ******************************************************************************/
#include <iomanip>
#include <numeric>

#include "common/parallel/vector_ets.h"
#include "common/utils/math.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/mpi/wrapper.h"

namespace kaminpar::dist {
void DistributedGraph::print() const {
    std::ostringstream buf;

    const int w = std::ceil(std::log10(global_n()));

    buf << "n=" << n() << " m=" << m() << " ghost_n=" << ghost_n() << " total_n=" << total_n() << "\n";
    buf << "--------------------------------------------------------------------------------\n";
    for (const NodeID u: all_nodes()) {
        const char u_prefix = is_owned_node(u) ? ' ' : '!';
        buf << u_prefix << "L" << std::setw(w) << u << " G" << std::setw(w) << local_to_global_node(u) << " W"
            << std::setw(w) << node_weight(u);

        if (is_owned_node(u)) {
            buf << " | ";
            for (const auto [e, v]: neighbors(u)) {
                const char v_prefix = is_owned_node(v) ? ' ' : '!';
                buf << v_prefix << "L" << std::setw(w) << v << " G" << std::setw(w) << local_to_global_node(v) << " EW"
                    << std::setw(w) << edge_weight(e) << "\t";
            }
            if (degree(u) == 0) {
                buf << "<empty>";
            }
        }
        buf << "\n";
    }
    buf << "--------------------------------------------------------------------------------\n";
    SLOG << buf.str();
}

namespace {
inline EdgeID degree_bucket(const EdgeID degree) {
    return (degree == 0) ? 0 : shm::math::floor_log2(degree) + 1;
}
} // namespace

void DistributedGraph::init_degree_buckets() {
    KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

    if (_sorted) {
        // @todo parallelize
        for (const NodeID u: nodes()) {
            ++_buckets[degree_bucket(degree(u)) + 1];
        }

        auto last_nonempty_bucket =
            std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
        _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
    } else {
        _buckets[1]        = n();
        _number_of_buckets = 1;
    }

    std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

void DistributedGraph::init_total_node_weight() {
    if (is_node_weighted()) {
        const auto begin_node_weights = _node_weights.begin();
        const auto end_node_weights   = begin_node_weights + static_cast<std::size_t>(n());

        _total_node_weight = shm::parallel::accumulate(begin_node_weights, end_node_weights, 0);
        _max_node_weight   = shm::parallel::max_element(begin_node_weights, end_node_weights);
    } else {
        _total_node_weight = n();
        _max_node_weight   = 1;
    }

    _global_total_node_weight = mpi::allreduce<GlobalNodeWeight>(_total_node_weight, MPI_SUM, communicator());
    _global_max_node_weight   = mpi::allreduce<GlobalNodeWeight>(_max_node_weight, MPI_MAX, communicator());
}

void DistributedGraph::init_communication_metrics() {
    const PEID size = mpi::get_comm_size(_communicator);

    tbb::enumerable_thread_specific<std::vector<EdgeID>> edge_cut_to_pe_ets{[&] {
        return std::vector<EdgeID>(size);
    }};
    tbb::enumerable_thread_specific<std::vector<EdgeID>> comm_vol_to_pe_ets{[&] {
        return std::vector<EdgeID>(size);
    }};

    pfor_nodes_range([&](const auto r) {
        auto&         edge_cut_to_pe = edge_cut_to_pe_ets.local();
        auto&         comm_vol_to_pe = comm_vol_to_pe_ets.local();
        shm::Marker<> counted_pe{static_cast<std::size_t>(size)};

        for (NodeID u = r.begin(); u < r.end(); ++u) {
            for (const auto v: adjacent_nodes(u)) {
                if (is_ghost_node(v)) {
                    const PEID owner = ghost_owner(v);
                    KASSERT(static_cast<std::size_t>(owner) < edge_cut_to_pe.size());
                    ++edge_cut_to_pe[owner];

                    if (!counted_pe.get(owner)) {
                        KASSERT(static_cast<std::size_t>(owner) < counted_pe.size());
                        counted_pe.set(owner);

                        KASSERT(static_cast<std::size_t>(owner) < comm_vol_to_pe.size());
                        ++comm_vol_to_pe[owner];
                    }
                }
            }
            counted_pe.reset();
        }
    });

    _edge_cut_to_pe.clear();
    _edge_cut_to_pe.resize(size);
    for (const auto& edge_cut_to_pe: edge_cut_to_pe_ets) { // PE x THREADS
        for (std::size_t i = 0; i < edge_cut_to_pe.size(); ++i) {
            _edge_cut_to_pe[i] += edge_cut_to_pe[i];
        }
    }

    _comm_vol_to_pe.clear();
    _comm_vol_to_pe.resize(size);
    for (const auto& comm_vol_to_pe: comm_vol_to_pe_ets) {
        for (std::size_t i = 0; i < comm_vol_to_pe.size(); ++i) {
            _comm_vol_to_pe[i] += comm_vol_to_pe[i];
        }
    }
}

void DistributedPartitionedGraph::init_block_weights() {
    parallel::vector_ets<BlockWeight> local_block_weights_ets(k());

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const auto& r) {
        auto& local_block_weights = local_block_weights_ets.local();
        for (NodeID u = r.begin(); u != r.end(); ++u) {
            local_block_weights[block(u)] += node_weight(u);
        }
    });
    auto local_block_weights = local_block_weights_ets.combine(std::plus{});

    scalable_vector<BlockWeight> global_block_weights_nonatomic(k());
    mpi::allreduce(local_block_weights.data(), global_block_weights_nonatomic.data(), k(), MPI_SUM, communicator());

    _block_weights.resize(k());
    pfor_blocks([&](const BlockID b) { _block_weights[b] = global_block_weights_nonatomic[b]; });
}

namespace graph {
void print_summary(const DistributedGraph& graph) {
    const auto global_n = graph.global_n();
    const auto global_m = graph.global_m();
    const auto [local_n_min, local_n_avg, local_n_max, local_n_sum] =
        mpi::gather_statistics(graph.n(), graph.communicator());
    const double local_n_imbalance = 1.0 * local_n_max / local_n_avg;
    const auto [local_m_min, local_m_avg, local_m_max, local_m_sum] =
        mpi::gather_statistics(graph.m(), graph.communicator());
    const double local_m_imbalance = 1.0 * local_m_max / local_m_avg;
    const auto [ghost_min, ghost_avg, ghost_max, ghost_sum] =
        mpi::gather_statistics(graph.ghost_n(), graph.communicator());
    const double ghost_imbalance = 1.0 * ghost_max / ghost_avg;

    const auto local_width =
        static_cast<std::streamsize>(std::log10(std::max({local_n_max, local_m_max, ghost_max})) + 1);

    LOG << "  Global number of nodes: " << global_n;
    LOG << "  Global number of edges: " << global_m;
    LOG << "  Local number of nodes:  [min=" << std::setw(local_width) << local_n_min << "|avg=" << std::fixed
        << std::setprecision(1) << std::setw(local_width) << local_n_avg << "|max=" << std::setw(local_width)
        << local_n_max << "|imbalance=" << std::fixed << std::setprecision(3) << std::setw(local_width)
        << local_n_imbalance << "]";
    LOG << "  Local number of edges:  [min=" << std::setw(local_width) << local_m_min << "|avg=" << std::fixed
        << std::setprecision(1) << std::setw(local_width) << local_m_avg << "|max=" << std::setw(local_width)
        << local_m_max << "|imbalance=" << std::fixed << std::setprecision(3) << std::setw(local_width)
        << local_m_imbalance << "]";
    LOG << "  Number of ghost nodes:  [min=" << std::setw(local_width) << ghost_min << "|avg=" << std::fixed
        << std::setprecision(1) << std::setw(local_width) << ghost_avg << "|max=" << std::setw(local_width) << ghost_max
        << "|imbalance=" << std::fixed << std::setprecision(3) << std::setw(local_width) << ghost_imbalance << "]";
}
} // namespace graph

namespace graph::debug {
SET_DEBUG(false);

namespace {
template <typename R>
bool all_equal(const R& r) {
    return std::adjacent_find(r.begin(), r.end(), std::not_equal_to{}) == r.end();
}
} // namespace

bool validate(const DistributedGraph& graph, const int root) {
    MPI_Comm comm = graph.communicator();
    mpi::barrier(comm);

    const auto [size, rank] = mpi::get_comm_info(comm);

    // check global n, global m
    DBG << "Checking global n, m";
    KASSERT(
        mpi::bcast(graph.global_n(), root, comm) == graph.global_n(), "inconsistent global number of nodes",
        assert::always);
    KASSERT(
        mpi::bcast(graph.global_m(), root, comm) == graph.global_m(), "inconsistent global number of edges",
        assert::always);

    // check global node distribution
    DBG << "Checking node distribution";
    KASSERT(
        static_cast<int>(graph.node_distribution().size()) == size + 1, "bad size of node distribution array",
        assert::always);
    KASSERT(graph.node_distribution().front() == 0u, "bad first entry of node distribution array", assert::always);
    KASSERT(
        graph.node_distribution().back() == graph.global_n(), "bad last entry of node distribution array",
        assert::always);
    for (PEID pe = 1; pe < size + 1; ++pe) {
        KASSERT(
            mpi::bcast(graph.node_distribution(pe), root, comm) == graph.node_distribution(pe),
            "inconsistent entry in node distribution array", assert::always);
        KASSERT(
            rank + 1 != pe || graph.node_distribution(pe) - graph.node_distribution(pe - 1) == graph.n(),
            "bad entry in node distribution array", assert::always);
    }

    // check global edge distribution
    DBG << "Checking edge distribution";
    KASSERT(
        static_cast<int>(graph.edge_distribution().size()) == size + 1, "bad size of edge distribution array",
        assert::always);
    KASSERT(graph.edge_distribution().front() == 0u, "bad first entry of edge distribution array", assert::always);
    KASSERT(
        graph.edge_distribution().back() == graph.global_m(), "bad last entry of edge distribution array",
        assert::always);
    for (PEID pe = 1; pe < size + 1; ++pe) {
        KASSERT(
            mpi::bcast(graph.edge_distribution(pe), root, comm) == graph.edge_distribution(pe),
            "inconsistent entry in edge distribution array", assert::always);
        KASSERT(
            rank + 1 != pe || graph.edge_distribution(pe) - graph.edge_distribution(pe - 1) == graph.m(),
            "bad entry in edge distribution array", assert::always);
    }

    // check that ghost nodes are actually ghost nodes
    DBG << "Checking ghost nodes";
    for (NodeID ghost_u: graph.ghost_nodes()) {
        KASSERT(graph.ghost_owner(ghost_u) != rank, "owner of ghost node should not be the same PE", assert::always);
    }

    // check node weight of ghost nodes
    DBG << "Checking node weights of ghost nodes";
    {
        struct GhostNodeWeightMessage {
            GlobalNodeID global_u;
            NodeWeight   weight;
        };

        mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeWeightMessage>(
            graph,
            [&](const NodeID u) -> GhostNodeWeightMessage {
                return {.global_u = graph.local_to_global_node(u), .weight = graph.node_weight(u)};
            },
            [&](const auto buffer, PEID) {
                for (const auto [global_u, weight]: buffer) {
                    KASSERT(
                        graph.contains_global_node(global_u),
                        "global node " << global_u << " has edge to this PE, but this PE does not know the node",
                        assert::always);
                    const NodeID local_u = graph.global_to_local_node(global_u);
                    KASSERT(
                        graph.node_weight(local_u) == weight,
                        "inconsistent weight for global node " << global_u << " / local node " << local_u,
                        assert::always);
                }
            });
    }

    // check that edges to ghost nodes exist in both directions
    DBG << "Checking edges to ghost nodes";
    {
        struct GhostNodeEdge {
            GlobalNodeID owned_node;
            GlobalNodeID ghost_node;
        };

        mpi::graph::sparse_alltoall_interface_to_ghost<GhostNodeEdge>(
            graph,
            [&](const NodeID u, const EdgeID, const NodeID v) -> GhostNodeEdge {
                return {.owned_node = graph.local_to_global_node(u), .ghost_node = graph.local_to_global_node(v)};
            },
            [&](const auto recv_buffer, const PEID pe) {
                for (const auto [ghost_node, owned_node]: recv_buffer) { // NOLINT: roles are swapped on receiving PE
                    KASSERT(
                        graph.contains_global_node(ghost_node),
                        "global node " << ghost_node << " has edge to this PE, but this PE does not know the node",
                        assert::always);
                    KASSERT(
                        graph.contains_global_node(owned_node),
                        "global node " << ghost_node << " has edge to global node " << owned_node
                                       << " which should be owned by this PE, but this PE does not know that node",
                        assert::always);

                    const NodeID local_owned_node = graph.global_to_local_node(owned_node);
                    const NodeID local_ghost_node = graph.global_to_local_node(ghost_node);

                    bool found = false;
                    for (const auto v: graph.adjacent_nodes(local_owned_node)) {
                        if (v == local_ghost_node) {
                            found = true;
                            break;
                        }
                    }
                    KASSERT(
                        found,
                        "local node " << local_owned_node << " (global node " << owned_node << ") "
                                      << "is expected to be adjacent to local node " << local_ghost_node
                                      << " (global node " << ghost_node << ") "
                                      << "due to an edge on PE " << pe << ", but is not",
                        assert::always);
                }
            });
    }

    // check that the graph is sorted if it claims that it is sorted
    DBG << "Checking degree buckets";
    if (graph.sorted()) {
        for (std::size_t bucket = 0; bucket < graph.number_of_buckets(); ++bucket) {
            if (graph.bucket_size(bucket) == 0) {
                continue;
            }

            KASSERT(
                graph.first_node_in_bucket(bucket) != graph.first_invalid_node_in_bucket(bucket),
                "bucket is empty, but graph data structure claims that it has size " << graph.bucket_size(bucket),
                assert::always);

            for (NodeID u = graph.first_node_in_bucket(bucket); u < graph.first_invalid_node_in_bucket(bucket); ++u) {
                const auto expected_bucket = shm::degree_bucket(graph.degree(u));
                KASSERT(
                    expected_bucket == bucket,
                    "node " << u << " with degree " << graph.degree(u) << " is expected to be in bucket "
                            << expected_bucket << ", but is in bucket " << bucket,
                    assert::always);
            }
        }
    }

    mpi::barrier(comm);
    return true;
}

bool validate_partition(const DistributedPartitionedGraph& p_graph) {
    MPI_Comm comm           = p_graph.communicator();
    const auto [size, rank] = mpi::get_comm_info(comm);

    {
        DBG << "Check that each PE knows the same block count";
        const auto recv_k = mpi::allgather(p_graph.k(), p_graph.communicator());
        KASSERT(all_equal(recv_k));
        mpi::barrier(comm);
    }

    {
        DBG << "Check that block IDs are OK";
        for (const NodeID u: p_graph.all_nodes()) {
            KASSERT(p_graph.block(u) < p_graph.k());
        }
    }

    {
        DBG << "Check that each PE has the same block weights";

        scalable_vector<BlockWeight> recv_block_weights;
        if (ROOT(rank)) {
            recv_block_weights.resize(size * p_graph.k());
        }
        const scalable_vector<BlockWeight> send_block_weights = p_graph.block_weights_copy();
        mpi::gather(
            send_block_weights.data(), static_cast<int>(p_graph.k()), recv_block_weights.data(),
            static_cast<int>(p_graph.k()), 0, comm);

        if (ROOT(rank)) {
            for (const BlockID b: p_graph.blocks()) {
                for (int pe = 0; pe < size; ++pe) {
                    const BlockWeight expected = recv_block_weights[b];
                    const BlockWeight actual   = recv_block_weights[p_graph.k() * pe + b];
                    KASSERT(
                        expected == actual, "for PE " << pe << ", block " << b << ": expected weight " << expected
                                                      << " (weight on root), got weight " << actual);
                }
            }
        }

        mpi::barrier(comm);
    }

    {
        DBG << "Check that block weights are actually correct";

        scalable_vector<BlockWeight> send_block_weights(p_graph.k());
        for (const NodeID u: p_graph.nodes()) {
            send_block_weights[p_graph.block(u)] += p_graph.node_weight(u);
        }
        scalable_vector<BlockWeight> recv_block_weights;
        if (ROOT(rank)) {
            recv_block_weights.resize(p_graph.k());
        }
        mpi::reduce(
            send_block_weights.data(), recv_block_weights.data(), static_cast<int>(p_graph.k()), MPI_SUM, 0, comm);
        if (ROOT(rank)) {
            for (const BlockID b: p_graph.blocks()) {
                KASSERT(p_graph.block_weight(b) == recv_block_weights[b]);
            }
        }

        mpi::barrier(comm);
    }

    {
        DBG << "Check whether the assignment of ghost nodes is consistent";

        // collect partition on root
        scalable_vector<BlockID> recv_partition;
        if (ROOT(rank)) {
            recv_partition.resize(p_graph.global_n());
        }

        const auto recvcounts = mpi::build_distribution_recvcounts(p_graph.node_distribution());
        const auto displs     = mpi::build_distribution_displs(p_graph.node_distribution());
        mpi::gatherv(
            p_graph.partition().data(), static_cast<int>(p_graph.n()), recv_partition.data(), recvcounts.data(),
            displs.data(), 0, comm);

        // next, each PE validates the block of its ghost nodes by sending them to root
        scalable_vector<std::uint64_t> send_buffer;
        send_buffer.reserve(p_graph.ghost_n() * 2);
        for (const NodeID ghost_u: p_graph.ghost_nodes()) {
            if (ROOT(rank)) { // root can validate locally
                KASSERT(p_graph.block(ghost_u) == recv_partition[p_graph.local_to_global_node(ghost_u)]);
            } else {
                send_buffer.push_back(p_graph.local_to_global_node(ghost_u));
                send_buffer.push_back(p_graph.block(ghost_u));
            }
        }

        // exchange messages and validate
        if (ROOT(rank)) {
            for (int pe = 1; pe < size; ++pe) { // recv from all but root
                const auto recv_buffer = mpi::probe_recv<std::uint64_t>(pe, 0, comm);

                // now validate received data
                for (std::size_t i = 0; i < recv_buffer.size(); i += 2) {
                    const auto global_u = static_cast<GlobalNodeID>(recv_buffer[i]);
                    const auto b        = static_cast<BlockID>(recv_buffer[i + 1]);
                    KASSERT(
                        recv_partition[global_u] == b,
                        "on PE " << pe << ": ghost node " << global_u << " is placed in block " << b
                                 << ", but on its owner PE, it is placed in block " << recv_partition[global_u]);
                }
            }
        } else {
            mpi::send(send_buffer.data(), send_buffer.size(), 0, 0, comm);
        }

        mpi::barrier(comm);
    }

    return true;
}
} // namespace graph::debug
} // namespace kaminpar::dist
