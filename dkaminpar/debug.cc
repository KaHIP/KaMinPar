/*******************************************************************************
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   16.05.2022
 * @brief:  Debug features.
 ******************************************************************************/
#include "dkaminpar/debug.h"

#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/wrapper.h"

#include "kaminpar/io.h"

#include "common/random.h"

namespace kaminpar::dist::debug {
namespace {
std::string create_basename(const Context& ctx, const int level) {
    return str::extract_basename(ctx.debug.graph_filename) + ".seed" + std::to_string(Random::seed) + ".k"
           + std::to_string(ctx.partition.k) + ".level" + std::to_string(level);
}
} // namespace

void save_partition(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level) {
    const auto        cut       = metrics::edge_cut(p_graph);
    const double      imbalance = metrics::imbalance(p_graph);
    const std::string filename  = create_basename(ctx, level) + ".part";

    if (mpi::get_comm_rank(p_graph.communicator()) == 0) {
        LOG_WARNING << "Writing partition of graph with " << p_graph.global_n() << " nodes and " << p_graph.global_m()
                    << " edges to " << filename;
        LOG_WARNING << "  Cut: " << cut;
        LOG_WARNING << "  Imbalance: " << imbalance;
    }

    io::partition::write(filename, p_graph);
}

void save_graph(const DistributedGraph& graph, const Context& ctx, const int level) {
    io::metis::write(create_basename(ctx, level) + ".graph", graph);
}

void save_graph(const shm::Graph& graph, const Context& ctx, const int level) {
    shm::io::metis::write(create_basename(ctx, level) + ".graph", graph);
}

void save_partitioned_graph(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level) {
    save_partition(p_graph, ctx, level);
    save_graph(p_graph.graph(), ctx, level);
}

void save_global_clustering(
    const scalable_vector<parallel::Atomic<GlobalNodeID>>& clustering, const Context& ctx, const int level
) {
    io::partition::write(create_basename(ctx, level) + ".clustering", clustering);
}
} // namespace kaminpar::dist::debug
