/*******************************************************************************
 * @file:   dist_io.h
 * @author: Daniel Seemaier
 * @date:   13.06.2023
 * @brief:  Distributed partition IO functions.
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

#include <dkaminpar/dkaminpar.h>

namespace kaminpar::dist::io::partition {
void write(const std::string &filename, const std::vector<BlockID> &partition);
}

