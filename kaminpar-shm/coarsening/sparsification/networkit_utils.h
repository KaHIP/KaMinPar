//
// Created by badger on 5/8/24.
//

#ifndef NETWORKIT_UTILS_H
#define NETWORKIT_UTILS_H
#include <networkit/graph/Graph.hpp>
#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm::sparsification::networkit_utils {
NetworKit::Graph toNetworKitGraph(const CSRGraph &g);
}

#endif //NETWORKIT_UTILS_H
