/*******************************************************************************
 * @file:   distributed_io.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Load distributed grpah from a single METIS file, node or edge
 * balanced.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi_wrapper.h"

#include <fstream>
#include <mpi.h>
#include <string>

namespace dkaminpar::io {
enum class DistributionType {
  NODE_BALANCED,
  EDGE_BALANCED,
};

DistributedGraph read_graph(const std::string &filename, DistributionType type, MPI_Comm comm = MPI_COMM_WORLD);

namespace metis {
DistributedGraph read_node_balanced(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD);
DistributedGraph read_edge_balanced(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD);
void write(const std::string &filename, const DistributedGraph &graph, bool write_node_weights = true,
           bool write_edge_weights = true);
} // namespace metis

namespace binary {
DistributedGraph read_node_balanced(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD);
DistributedGraph read_edge_balanced(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD);
} // namespace binary

namespace partition {
template <typename Container> void write(const std::string &filename, const Container &partition) {
  mpi::sequentially([&] {
    std::ofstream out(filename, std::ios_base::out | std::ios_base::app);
    for (const BlockID &b : partition) {
      out << b << "\n";
    }
  });
}
} // namespace partition
} // namespace dkaminpar::io