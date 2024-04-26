/*******************************************************************************
 * IO utilities for the compressed graph binary.
 *
 * @file:   shm_compressed_graph_binary.h
 * @author: Daniel Salwasser
 * @date:   12.12.2023
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-shm/datastructures/compressed_graph.h"

namespace kaminpar::shm::io::compressed_binary {

//! Magic number to identify a compressed graph binary file.
constexpr std::uint64_t kMagicNumber = 0x434F4D5052455353;

/*!
 * Writes a compressed graph to a file in binary format.
 *
 * @param filename The name of the file to write to.
 * @param graph The compressed graph to write.
 */
void write(const std::string &filename, const CompressedGraph &graph);

/*!
 * Reads a compressed graph from a file with binary format. If the paramters of the compressed graph
 * stored in the file do not match with this build, exit is called.
 *
 * @param filename The name of the file to read from.
 * @return The read compressed graph.
 */
CompressedGraph read(const std::string &filename);

/*!
 * Checks whether a graph is stored in compressed binary format.
 *
 * @param filename The name of the file to check.
 * @return Whether the graph is stored in compressed format.
 */
bool is_compressed(const std::string &filename);

} // namespace kaminpar::shm::io::compressed_binary
