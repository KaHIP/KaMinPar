/*******************************************************************************
 * Sequential METIS parser.
 *
 * @file:   metis_parser.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm::io::metis {

/**
 * Reads a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
CSRGraph csr_read(const std::string &filename, const bool sorted);

/*!
 * Reads and compresses a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
CompressedGraph compress_read(const std::string &filename, const bool sorted);

/*!
 * Writes a graph to a file in METIS format.
 *
 * @param filename The name of the file in which to store the graph.
 * @param graph The graph to store.
 */
void write(const std::string &filename, const Graph &graph);

} // namespace kaminpar::shm::io::metis
