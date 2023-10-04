/*******************************************************************************
 * IO functions for the distributed partitioner.
 *
 * @file:   dist_io.h
 * @author: Daniel Seemaier
 * @date:   13.06.2023
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::io::partition {
void write(const std::string &filename, const std::vector<BlockID> &partition);
}
