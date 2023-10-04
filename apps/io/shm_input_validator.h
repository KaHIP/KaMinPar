/*******************************************************************************
 * Validator for undirected input graphs.
 *
 * @file:   shm_input_validator.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {
void validate_undirected_graph(
    const StaticArray<EdgeID> &nodes,
    const StaticArray<NodeID> &edges,
    const StaticArray<NodeWeight> &node_weights,
    const StaticArray<EdgeWeight> &edge_weights
);
} // namespace kaminpar::shm
