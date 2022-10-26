/*******************************************************************************
 * @file:   io.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Graph and partition IO functions.
 ******************************************************************************/
#pragma once

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"

#include "common/assert.h"
#include "common/io/metis_parser.h"
#include "common/io/mmap_toker.h"

namespace kaminpar::shm::io {
namespace metis {
struct Statistics {
    std::uint64_t total_node_weight  = 0;
    std::uint64_t total_edge_weight  = 0;
    bool          has_isolated_nodes = false;
};

template <bool checked>
Graph read(const std::string& filename, bool ignore_node_weights = false, bool ignore_edge_weights = false);

template <bool checked>
Statistics read(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights
);

void write(const std::string& filename, const Graph& graph, const std::string& comment = "");
} // namespace metis

namespace partition {
void write(const std::string& filename, const std::vector<BlockID>& partition);
void write(const std::string& filename, const PartitionedGraph& p_graph);
void write(const std::string& filename, const StaticArray<BlockID>& partition, const StaticArray<NodeID>& permutation);
void write(const std::string& filename, const PartitionedGraph& p_graph, const StaticArray<NodeID>& permutation);

template <typename Container = std::vector<BlockID>>
Container read(const std::string& filename) {
    using namespace kaminpar::io;

    MappedFileToker<>    toker(filename);
    std::vector<BlockID> partition;
    while (toker.valid_position()) {
        partition.push_back(toker.scan_uint());
        toker.consume_char('\n');
    }

    if constexpr (std::is_same_v<Container, std::vector<BlockID>>) {
        return partition;
    } else {
        Container copy(partition.size());
        std::copy(partition.begin(), partition.end(), copy.begin());
        return copy;
    }
}
} // namespace partition
} // namespace kaminpar::shm::io
