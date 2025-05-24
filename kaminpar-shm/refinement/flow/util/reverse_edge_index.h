#pragma once

#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

[[nodiscard]] StaticArray<EdgeID> compute_reverse_edge_index(const CSRGraph &graph);

namespace debug {

[[nodiscard]] bool
is_valid_reverse_edge_index(const CSRGraph &graph, std::span<const NodeID> reverse_edges);

}

} // namespace kaminpar::shm
