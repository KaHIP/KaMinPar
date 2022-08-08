/*******************************************************************************
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   16.05.2022
 * @brief:  Debug features.
 ******************************************************************************/
#include "dkaminpar/debug.h"

#include "dkaminpar/io.h"

namespace kaminpar::dist::debug {
namespace {
std::string create_basename(const Context& ctx, const int level) {
    return str::extract_basename(ctx.graph_filename) + ".seed" + std::to_string(ctx.seed) + ".k"
           + std::to_string(ctx.partition.k) + ".level" + std::to_string(level);
}
} // namespace

void save_partition(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level) {
    std::vector<BlockID> partition(p_graph.n());
    std::copy_n(p_graph.partition().begin(), p_graph.n(), partition.begin());
    io::partition::write(create_basename(ctx, level) + ".part", partition);
}

void save_graph(const DistributedGraph& graph, const Context& ctx, const int level) {
    io::metis::write(create_basename(ctx, level) + ".graph", graph);
}

void save_partitioned_graph(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level) {
    save_partition(p_graph, ctx, level);
    save_graph(p_graph.graph(), ctx, level);
}

void save_global_clustering(
    const scalable_vector<parallel::Atomic<GlobalNodeID>>& clustering, const Context& ctx, const int level) {
    io::partition::write(create_basename(ctx, level) + ".clustering", clustering);
}
} // namespace kaminpar::dist::debug
