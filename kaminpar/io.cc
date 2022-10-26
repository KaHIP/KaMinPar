/*******************************************************************************
 * @file:   io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Graph and partition IO functions.
 ******************************************************************************/
#include "kaminpar/io.h"

#include "common/timer.h"

namespace kaminpar::shm::io {
namespace metis {
void write_file(
    std::ofstream& out, const StaticArray<EdgeID>& nodes, const StaticArray<NodeID>& edges,
    const StaticArray<NodeWeight>& node_weights, const StaticArray<EdgeWeight>& edge_weights, const std::string& comment
) {
    const bool write_node_weights = !node_weights.empty();
    const bool write_edge_weights = !edge_weights.empty();

    if (!comment.empty()) {
        out << "% " << comment << "\n";
    }

    // header
    out << nodes.size() - 1 << " " << edges.size() / 2;
    if (write_node_weights || write_edge_weights) {
        out << " " << static_cast<int>(write_node_weights) << static_cast<int>(write_edge_weights);
    }
    out << "\n";

    // content
    for (NodeID u = 0; u < nodes.size() - 1; ++u) {
        if (write_node_weights) {
            out << node_weights[u] << " ";
        }
        for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
            out << edges[e] + 1 << " ";
            if (write_edge_weights) {
                out << edge_weights[e] << " ";
            }
        }
        out << "\n";
    }
}

void write_file(
    const std::string& filename, const StaticArray<EdgeID>& nodes, const StaticArray<NodeID>& edges,
    const StaticArray<NodeWeight>& node_weights, const StaticArray<EdgeWeight>& edge_weights, const std::string& comment
) {
    std::ofstream out(filename);
    if (!out) {
        FATAL_PERROR << "Error while opening " << filename;
    }
    write_file(out, nodes, edges, node_weights, edge_weights, comment);
}
} // namespace metis

//
// Public Metis functions
//

namespace metis {
template <bool checked>
Statistics read(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights
) {
    using namespace kaminpar::io::metis;

    bool store_node_weights = false;
    bool store_edge_weights = false;

    NodeID u = 0;
    EdgeID e = 0;

    Statistics stats;

    parse<false>(
        filename,
        [&](const auto& format) {
            if constexpr (checked) {
                if (format.number_of_nodes >= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max())) {
                    LOG_ERROR << "number of nodes is too large for the node ID type";
                    std::exit(1);
                }
                if (format.number_of_edges >= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max())) {
                    LOG_ERROR << "number of edges is too large for the edge ID type";
                    std::exit(1);
                }
            } else {
                KASSERT(
                    format.number_of_nodes <= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
                    "number of nodes is too large for the node ID type"
                );
                KASSERT(
                    format.number_of_edges <= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
                    "number of edges is too large for the edge ID type"
                );
            }

            store_node_weights = format.has_node_weights;
            store_edge_weights = format.has_edge_weights;
            nodes.resize(format.number_of_nodes + 1);
            edges.resize(format.number_of_edges * 2);
            if (store_node_weights) {
                node_weights.resize(format.number_of_nodes);
            }
            if (store_edge_weights) {
                edge_weights.resize(format.number_of_edges * 2);
            }
        },
        [&](const std::uint64_t weight) {
            if constexpr (checked) {
                if (weight > static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max())) {
                    LOG_ERROR << "node weight is too large for the node weight type";
                    std::exit(1);
                }
                if (weight <= 0) {
                    LOG_ERROR << "zero node weights are not supported";
                    std::exit(1);
                }
            } else {
                KASSERT(
                    weight <= static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max()),
                    "node weight is too large for the node weight type"
                );
                KASSERT(weight > 0u, "zero node weights are not supported");
            }

            stats.total_node_weight += weight;
            stats.has_isolated_nodes |= (u > 0 && nodes[u - 1] == e);

            if (store_node_weights) {
                node_weights[u] = static_cast<NodeWeight>(weight);
            }
            nodes[u] = e;
            ++u;
        },
        [&](const std::uint64_t weight, const std::uint64_t v) {
            if constexpr (checked) {
                if (weight > static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
                    LOG_ERROR << "edge weight is too large for the edge weight type";
                    std::exit(1);
                }
                if (weight <= 0) {
                    LOG_ERROR << "zero edge weights are not supported";
                    std::exit(1);
                }
                if (v + 1 >= nodes.size()) {
                    LOG_ERROR << "neighbor " << v + 1 << " of nodes " << u + 1 << " is out of bounds";
                    std::exit(1);
                }
                if (v + 1 == u) {
                    LOG_ERROR << "detected self-loop on node " << v + 1 << ", which is not allowed";
                    std::exit(1);
                }
            } else {
                KASSERT(
                    weight <= static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()),
                    "edge weight is too large for the edge weight type"
                );
                KASSERT(weight > 0u, "zero edge weights are not supported");
                KASSERT(v + 1 < nodes.size(), "neighbor out of bounds");
                KASSERT(u != v + 1, "detected illegal self-loop");
            }

            stats.total_edge_weight += weight;
            if (store_edge_weights) {
                edge_weights[e] = static_cast<EdgeWeight>(weight);
            }
            edges[e] = static_cast<NodeID>(v);
            ++e;
        }
    );
    nodes[u] = e;

    if constexpr (checked) {
        if (stats.total_node_weight > static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max())) {
            LOG_ERROR << "total node weight does not fit into the node weight type";
            std::exit(1);
        }
        if (stats.total_edge_weight > static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
            LOG_ERROR << "total edge weight does not fit into the edge weight type";
            std::exit(1);
        }
    } else {
        KASSERT(
            stats.total_node_weight <= static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max()),
            "total node weight does not fit into the node weight type"
        );
        KASSERT(
            stats.total_edge_weight <= static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()),
            "total edge weight does not fit into the edge weight type"
        );
    }

    // only keep weights if the graph is really weighted
    const bool unit_node_weights = stats.total_node_weight + 1 == nodes.size();
    const bool unit_edge_weights = stats.total_edge_weight == edges.size();
    if (unit_node_weights) {
        node_weights.free();
    }
    if (unit_edge_weights) {
        edge_weights.free();
    }

    return stats;
}

template Statistics read<false>(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights
);

template Statistics read<true>(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights
);

template <bool checked>
Graph read(const std::string& filename, bool ignore_node_weights, bool ignore_edge_weights) {
    StaticArray<EdgeID>     nodes;
    StaticArray<NodeID>     edges;
    StaticArray<NodeWeight> node_weights;
    StaticArray<EdgeWeight> edge_weights;

    metis::read<checked>(filename, nodes, edges, node_weights, edge_weights);

    if (ignore_node_weights) {
        node_weights.free();
    }
    if (ignore_edge_weights) {
        edge_weights.free();
    }

    return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}

template Graph read<false>(const std::string& filename, bool ignore_node_weights, bool ignore_edge_weights);

template Graph read<true>(const std::string& filename, bool ignore_node_weights, bool ignore_edge_weights);

void write(const std::string& filename, const Graph& graph, const std::string& comment) {
    metis::write_file(
        filename, graph.raw_nodes(), graph.raw_edges(), graph.raw_node_weights(), graph.raw_edge_weights(), comment
    );
}
} // namespace metis

//
// Partition
//

namespace partition {
void write(const std::string& filename, const StaticArray<BlockID>& partition) {
    std::ofstream out(filename);
    for (const BlockID block: partition) {
        out << block << "\n";
    }
}

void write(const std::string& filename, const PartitionedGraph& p_graph) {
    write(filename, p_graph.partition());
}

void write(const std::string& filename, const StaticArray<BlockID>& partition, const StaticArray<NodeID>& permutation) {
    std::ofstream out(filename);
    for (const NodeID u: permutation) {
        out << partition[u] << "\n";
    }
}

void write(const std::string& filename, const PartitionedGraph& p_graph, const StaticArray<NodeID>& permutation) {
    write(filename, p_graph.partition(), permutation);
}
} // namespace partition
} // namespace kaminpar::shm::io
