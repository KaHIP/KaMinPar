/*******************************************************************************
 * Sequential and parallel ParHiP parser for distributed compressed graphs.
 *
 * @file:   dist_parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   11.05.2024
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::io::parhip {

/*!
 * Reads a distributed graph that is stored in a file with ParHiP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @param distribution How the graph is distributed among the processes.
 * @param comm The group of processes that read the distributed graph.
 * @return The graph that is stored in the file.
 */
DistributedCSRGraph csr_read(
    const std::string &filename,
    const GraphDistribution distribution,
    const bool sorted,
    const MPI_Comm comm
);

/*!
 * Reads and compresses a distributed graph that is stored in a file with ParHiP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @param distribution How the graph is distributed among the processes.
 * @param comm The group of processes that read and compress the distributed graph.
 * @return The graph that is stored in the file.
 */
DistributedCompressedGraph compressed_read(
    const std::string &filename,
    const GraphDistribution distribution,
    const bool sorted,
    const MPI_Comm comm
);

} // namespace kaminpar::dist::io::parhip
