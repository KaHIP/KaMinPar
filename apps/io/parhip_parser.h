/*******************************************************************************
 * Sequential ParHiP parser.
 *
 * @file:   parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-shm/datastructures/csr_graph.h"

namespace kaminpar::shm::io::parhip {

CSRGraph csr_read(const std::string &filename, const bool sorted);

} // namespace kaminpar::shm::io::parhip
