#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification::utils {

StaticArray<EdgeID> sort_by_traget(const CSRGraph &g);


}
