/*******************************************************************************
 * @file:   bfs_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner based on breath-first searches.
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/bfs_bipartitioner.h"

namespace kaminpar::shm::ip {
template class bfs::BfsBipartitioner<bfs::alternating>;
template class bfs::BfsBipartitioner<bfs::lighter>;
template class bfs::BfsBipartitioner<bfs::sequential>;
template class bfs::BfsBipartitioner<bfs::longer_queue>;
template class bfs::BfsBipartitioner<bfs::shorter_queue>;
} // namespace kaminpar::shm::ip
