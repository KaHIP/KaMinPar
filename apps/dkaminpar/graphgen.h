/*******************************************************************************);
 * @file:   dkaminpar_graphgen.h
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
DistributedGraph generate(const std::string& properties);
std::string      generate_filename(const std::string& properties);
} // namespace kaminpar::dist
