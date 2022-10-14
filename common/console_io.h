/*******************************************************************************
 * @file:   console_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#pragma once

#include <string>

#include <kassert/kassert.hpp>

#include "common/assert.h"
#include "common/logger.h"

namespace kaminpar::cio {
void print_delimiter();
void print_kaminpar_banner();
void print_dkaminpar_banner();
void print_banner(const std::string& title);

template <
    typename NodeID, typename EdgeID, typename NodeWeight, typename EdgeWeight, typename LocalNodeWeight = NodeWeight,
    typename LocalEdgeWeight = EdgeWeight>
void print_build_identifier(const std::string& commit, const std::string& hostname) {
    LOG << "Current commit hash:          " << (commit.empty() ? "<not available>" : commit);
    std::string assertion_level_name = "always";
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_LIGHT) {
        assertion_level_name += "+light";
    }
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_NORMAL) {
        assertion_level_name += "+normal";
    }
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_HEAVY) {
        assertion_level_name += "+heavy";
    }
    LOG << "Assertion level:              " << assertion_level_name;
#ifdef KAMINPAR_ENABLE_STATISTICS
    LOG << "Statistics:                   enabled";
#else
    LOG << "Statistics:                   disabled";
#endif
    LOG << "Data type sizes:";
    LOG << "  Nodes IDs: " << sizeof(NodeID) << " bytes | Node weights (Local): " << sizeof(LocalNodeWeight)
        << " bytes | Node weights (IP): " << sizeof(NodeWeight);
    LOG << "  Edges IDs: " << sizeof(EdgeID) << " bytes | Edge weights (Local): " << sizeof(LocalEdgeWeight)
        << " bytes | Edge weights (IP): " << sizeof(EdgeWeight);
    LOG << "Built on:                     " << (hostname.empty() ? "<not available>" : hostname);
    LOG << "################################################################################";
}

} // namespace kaminpar::cio
