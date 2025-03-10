/*******************************************************************************
 * Validator for undirected input graphs.
 *
 * @file:   graph_validator.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

void validate_undirected_graph(const Graph &graph);

} // namespace kaminpar::shm
