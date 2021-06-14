#include "initial_partitioning/bfs_bipartitioner.h"

namespace kaminpar {
template class bfs::BfsBipartitioner<bfs::alternating>;
template class bfs::BfsBipartitioner<bfs::lighter>;
template class bfs::BfsBipartitioner<bfs::sequential>;
template class bfs::BfsBipartitioner<bfs::longer_queue>;
template class bfs::BfsBipartitioner<bfs::shorter_queue>;
} // namespace kaminpar