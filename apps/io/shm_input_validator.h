/*******************************************************************************
 * @file:   shm_input_validator.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 * @brief:  Validator for undirected input graphs.
 ******************************************************************************/
#pragma once

#include "kaminpar/definitions.h"

#include "common/datastructures/static_array.h"

namespace kaminpar::shm {
void validate_undirected_graph(const StaticArray<EdgeID> &nodes,
                               const StaticArray<NodeID> &edges,
                               const StaticArray<NodeWeight> &node_weights,
                               const StaticArray<EdgeWeight> &edge_weights);
} // namespace kaminpar::shm
