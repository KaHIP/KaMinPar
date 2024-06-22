/*******************************************************************************
 * Sequential METIS parser for distributed graphs.
 *
 * @file:   dist_metis_parser.h
 * @author: Daniel Salwasser
 * @date:   22.06.2024
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"

namespace kaminpar::dist::io::metis {

/*!
 * Reads and compresses a graph that is stored in a file with METIS format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @param may_dismiss Whether to abort the compression when it is determined that the compressed
 * graph uses more memory than the uncompressed graph.
 * @return The graph that is stored in the file, or nothing if the graph was dismissed.
 */
DistributedCompressedGraph
compress_read(const std::string &filename, const bool sorted, const MPI_Comm comm);

} // namespace kaminpar::dist::io::metis
