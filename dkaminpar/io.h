/*******************************************************************************
 * @file:   io.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Load distributed grpah from a single METIS file, node or edge
 * balanced.
 ******************************************************************************/
#pragma once

#include <fstream>
#include <string>

#include <mpi.h>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/mpi/wrapper.h"

#include "kaminpar/io.h"

#include "common/io/mmap_toker.h"

namespace kaminpar::dist::io {
enum class DistributionType {
    NODE_BALANCED,
    EDGE_BALANCED,
};

DistributedGraph read_graph(const std::string& filename, DistributionType type, MPI_Comm comm = MPI_COMM_WORLD);

namespace metis {
DistributedGraph read_node_balanced(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
DistributedGraph read_edge_balanced(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
void             write(
                const std::string& filename, const DistributedGraph& graph, bool write_node_weights = true,
                bool write_edge_weights = true
            );
} // namespace metis

namespace binary {
DistributedGraph read_node_balanced(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
DistributedGraph read_edge_balanced(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
} // namespace binary

namespace partition {
template <typename Container>
Container read(const std::string& filename, const NodeID n, MPI_Comm comm = MPI_COMM_WORLD) {
    using namespace kaminpar::io;

    const GlobalNodeID offset = mpi::exscan(static_cast<GlobalNodeID>(n), MPI_SUM, comm);

    MappedFileToker      toker(filename);
    GlobalNodeID         current = 0;
    std::vector<BlockID> partition;

    while (toker.valid_position()) {
        if (current >= offset + n) {
            break;
        } else if (current >= offset) {
            partition.push_back(toker.scan_uint<std::uint64_t>());
        } else {
            toker.scan_uint<std::uint64_t>();
        }
        toker.consume_char('\n');
        ++current;
    }

    if constexpr (std::is_same_v<Container, std::vector<BlockID>>) {
        return partition;
    } else {
        Container copy(partition.size());
        std::copy(partition.begin(), partition.end(), copy.begin());
        return copy;
    }
}

template <typename Container>
void write(const std::string& filename, const Container& partition) {
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) { // clear file
        std::ofstream tmp(filename);
    }
    mpi::barrier(MPI_COMM_WORLD);

    mpi::sequentially(
        [&](PEID) {
            std::ofstream out(filename, std::ios_base::out | std::ios_base::app);
            for (const auto& b: partition) {
                out << b << "\n";
            }
        },
        MPI_COMM_WORLD
    );
}
} // namespace partition
} // namespace kaminpar::dist::io
