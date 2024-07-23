/*******************************************************************************
 * Sequential and parallel ParHiP parser.
 *
 * @file:   parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::io::parhip {

/**
 * Reads a graph that is stored in a file with ParHIP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
CSRGraph csr_read(const std::string &filename, const bool sorted);

/*!
 * Reads and compresses a graph that is stored in a file in ParHiP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
CompressedGraph compressed_read(const std::string &filename, const bool sorted);

/*!
 * Reads and compresses a graph that is stored in a file in ParHiP format in parallel.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
CompressedGraph compressed_read_parallel(const std::string &filename, const NodeOrdering ordering);

} // namespace kaminpar::shm::io::parhip
