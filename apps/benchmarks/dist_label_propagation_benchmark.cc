// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/graphutils/rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/timer.h"

#include "common/datastructures/rating_map.h"
#include "common/logger.h"
#include "common/math.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace kaminpar::dist {
struct UnorderedRatingMap {
    EdgeWeight& operator[](const GlobalNodeID key) {
        return map[key];
    }

    [[nodiscard]] auto& entries() {
        return map;
    }

    void clear() {
        map.clear();
    }

    std::size_t capacity() const {
        return std::numeric_limits<std::size_t>::max();
    }

    void resize(const std::size_t /* capacity */) {}

    std::unordered_map<GlobalNodeID, EdgeWeight> map{};
};

std::vector<GlobalNodeID> naive_label_propagation(
    const DistributedGraph& graph, const int num_iterations, const int num_chunks, const GlobalNodeWeight max_cluster_weight
) {
    SET_DEBUG(false);

    START_TIMER("Allocate clustering[]");
    std::vector<GlobalNodeID> clustering(graph.total_n());
    STOP_TIMER();
    START_TIMER("Allocate cluster_weights[]");
    using ClusterWeightsMap = growt::GlobalNodeIDMap<GlobalNodeWeight>;
    ClusterWeightsMap                                               cluster_weights(graph.total_n());
    tbb::enumerable_thread_specific<ClusterWeightsMap::handle_type> cluster_weights_ets([&] {
        return cluster_weights.get_handle();
    });
    STOP_TIMER();
    START_TIMER("Allocate changed_label[]");
    std::vector<std::uint8_t> changed_label(graph.n());
    STOP_TIMER();
    START_TIMER("Allocate rating map");
    using ClusterRatingMap = RatingMap<GlobalEdgeWeight, GlobalNodeID, UnorderedRatingMap>;
    tbb::enumerable_thread_specific<ClusterRatingMap> rating_maps_ets([&] { return ClusterRatingMap(graph.total_n()); }
    );
    STOP_TIMER();

    START_TIMER("Initialize clustering[]");
    tbb::parallel_for<NodeID>(0, graph.total_n(), [&](const NodeID u) {
        clustering[u] = graph.local_to_global_node(u);
    });
    STOP_TIMER();
    START_TIMER("Initialize cluster_weights[]");
    tbb::parallel_for<NodeID>(0, graph.total_n(), [&](const NodeID u) {
        const GlobalNodeID global_u               = graph.local_to_global_node(u);
        auto&              handle                 = cluster_weights_ets.local();
        [[maybe_unused]] const auto [it, success] = handle.insert(global_u + 1, graph.node_weight(u));
        KASSERT(success, "could not initialize cluster for node " << u);
    });
    STOP_TIMER();

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const auto [from, to] = math::compute_local_range<NodeID>(graph.n(), num_chunks, chunk);
            // Perform label propagation
            START_TIMER("Chunk iteration");
            DBG << " -> assign labels [" << from << ", " << to << ") ...";
            tbb::parallel_for<NodeID>(from, to, [&](const NodeID u) {
                auto& rating_map = rating_maps_ets.local();
                auto& handle     = cluster_weights_ets.local();

                const GlobalNodeID     cluster_u = __atomic_load_n(&clustering[u], __ATOMIC_RELAXED);
                const GlobalNodeWeight weight_u  = graph.node_weight(u);

                auto action = [&](auto& map) {
                    for (const auto [e, v]: graph.neighbors(u)) {
                        const GlobalNodeID cluster_v = __atomic_load_n(&clustering[v], __ATOMIC_RELAXED);
                        map[cluster_v] += graph.edge_weight(e);
                    }

                    GlobalNodeID     best_cluster = cluster_u;
                    GlobalEdgeWeight best_rating  = 0;

                    for (const auto [current_cluster, current_rating]: map.entries()) {
                        if (current_rating <= best_rating) {
                            continue;
                        }

                        auto it = handle.find(current_cluster + 1);
                        KASSERT(
                            it != handle.end(), "uninitialized cluster " << current_cluster + 1
                                                                         << " while assigning node " << u
                                                                         << " in cluster " << cluster_u
                        );
                        const GlobalNodeWeight current_weight = (*it).second;

                        // DBG << V(best_rating) << V(current_rating) << V(current_weight) << V(weight_u)
                        //<< V(current_weight) << V(max_cluster_weight);

                        if (current_weight + weight_u <= max_cluster_weight) {
                            best_cluster = current_cluster;
                            best_rating  = current_rating;
                        }
                    }

                    map.clear();
                    return best_cluster;
                };

                rating_map.update_upper_bound_size(graph.degree(u));
                const auto new_label = rating_map.run_with_map(action, action);

                if (new_label != cluster_u) {
                    // DBG << "Move " << u << " from " << cluster_u << " to " << new_label;

                    [[maybe_unused]] const auto [it1, success1] = handle.update(
                        cluster_u + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, -weight_u
                    );
                    KASSERT(success1);
                    [[maybe_unused]] const auto [it2, success2] = handle.update(
                        new_label + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, weight_u
                    );
                    KASSERT(success2);
                    __atomic_store_n(&clustering[u], new_label, __ATOMIC_RELAXED);
                    changed_label[u] = 1;
                }
            });
            mpi::barrier(MPI_COMM_WORLD);
            STOP_TIMER();

            DBG << "Moved " << std::count_if(changed_label.begin(), changed_label.end(), [&](const auto entry) {
                return entry != 0;
            });

            // Synchronize labels
            START_TIMER("Synchronize labels");
            DBG << " -> synchronize labels [" << from << ", " << to << ") ...";

            struct ChangedLabelMessage {
                NodeID       local_node;
                GlobalNodeID new_label;
            };

            mpi::graph::sparse_alltoall_interface_to_pe<ChangedLabelMessage>(
                graph, from, to,
                [&](const NodeID u) {
                    KASSERT(u < changed_label.size());
                    return changed_label[u];
                },
                [&](const NodeID u) -> ChangedLabelMessage {
                    return {u, clustering[u]};
                },
                [&](const auto& buffer, const PEID pe) {
                    tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                        const auto [local_node_on_pe, new_label] = buffer[i];
                        const GlobalNodeID global_node           = graph.offset_n(pe) + local_node_on_pe;
                        const NodeID       local_node            = graph.global_to_local_node(global_node);
                        const NodeWeight   local_node_weight     = graph.node_weight(local_node);
                        const GlobalNodeID old_label             = clustering[local_node];

                        auto& handle                                = cluster_weights_ets.local();
                        [[maybe_unused]] const auto [it1, success1] = handle.update(
                            old_label + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, -local_node_weight
                        );
                        KASSERT(success1);
                        handle.insert_or_update(
                            new_label + 1, local_node_weight, [](auto& lhs, const auto rhs) { return lhs += rhs; },
                            local_node_weight
                        );
                        clustering[local_node] = new_label;
                    });
                }
            );
            mpi::barrier(MPI_COMM_WORLD);
            STOP_TIMER();

            START_TIMER("Reset changed_label[]");
            tbb::parallel_for<NodeID>(from, to, [&](const NodeID u) { changed_label[u] = 0; });
            mpi::barrier(MPI_COMM_WORLD);
            STOP_TIMER();
        }
    }

    return clustering;
}
}; // namespace kaminpar::dist

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

    std::string graph_filename = "";
    std::string out_filename   = "";
    int         num_threads    = 1;
    int num_chunks = 1;
    int         num_iterations = 1;

    CLI::App app("Distributed Label Propagation Benchmark");
    app.add_option("-G,--graph", graph_filename, "Input graph")->required();
    app.add_option("-o,--output", out_filename, "Name of the clustering file.");
    app.add_option("-t,--threads", num_threads, "Number of threads");
    app.add_option("-n,--num-iterations", num_iterations, "Number of LP iterations");
    app.add_option("-c,--num-chunks", num_chunks, "Number of LP chunks");
    CLI11_PARSE(app, argc, argv);

    auto gc = init_parallelism(num_threads);
    omp_set_num_threads(num_threads);
    init_numa();

    /*****
     * Load graph
     */
    LOG << "Reading graph from " << graph_filename << " ...";
    DISABLE_TIMERS();
    auto graph = dist::io::read_graph(graph_filename, dist::io::DistributionType::NODE_BALANCED);
    ENABLE_TIMERS();
    LOG << "n=" << graph.global_n() << " m=" << graph.global_m();

    /****
     * Run label propagation
     */
    const GlobalNodeWeight max_cluster_weight = 0.03 * graph.global_n() / 2;
    LOG << "Running label propagation ...";
    LOG << " -> iterations: " << num_iterations;
    LOG << " -> max cluster weight: " << max_cluster_weight;

    START_TIMER("Label propagation");
    auto clustering = naive_label_propagation(graph, num_iterations, num_chunks, max_cluster_weight);
    STOP_TIMER();

    // Write the clustering to a text file
    if (!out_filename.empty()) {
        LOG << "Writing clustering to " << out_filename << " ...";
        if (rank == 0) {
            std::ofstream tmp(out_filename);
        }
        mpi::sequentially(
            [&](PEID) {
                std::ofstream out(out_filename, std::ios_base::out | std::ios_base::app);
                for (const NodeID u: graph.nodes()) {
                    out << clustering[u] << "\n";
                }
            },
            MPI_COMM_WORLD
        );
    }

    /*****
     * Clean up and print timer tree
     */
    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER();
    finalize_distributed_timer(GLOBAL_TIMER);
    if (rank == 0) {
        Timer::global().print_human_readable(std::cout);
    }
    return MPI_Finalize();
}
