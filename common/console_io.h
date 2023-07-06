/*******************************************************************************
 * Helper functions for console IO.
 *
 * @file:   console_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <ostream>
#include <string>

#include <kassert/kassert.hpp>

#include "common/assertion_levels.h"
#include "common/environment.h"
#include "common/logger.h"

namespace kaminpar::cio {
void print_delimiter(const std::string &caption = "", char ch = '#');
void print_kaminpar_banner();
void print_dkaminpar_banner();
void print_banner(const std::string &title);
void print_build_identifier();

template <typename NodeID, typename EdgeID, typename NodeWeight, typename EdgeWeight>
void print_build_datatypes() {
  LOG << "Data type sizes:";
  LOG << "  Nodes IDs: " << sizeof(NodeID) << " bytes | Node weights: " << sizeof(NodeWeight)
      << " bytes";
  LOG << "  Edges IDs: " << sizeof(EdgeID) << " bytes | Edge weights: " << sizeof(EdgeWeight)
      << " bytes";
}

template <
    typename NodeID,
    typename EdgeID,
    typename LocalNodeWeight,
    typename LocalEdgeWeight,
    typename IPNodeWeight,
    typename IPEdgeWeight>
void print_build_datatypes() {
  LOG << "Data type sizes:";
  LOG << "  Nodes IDs: " << sizeof(NodeID)
      << " bytes | Node weights (Local): " << sizeof(LocalNodeWeight)
      << " bytes | Node weights (IP): " << sizeof(IPNodeWeight) << " bytes";
  LOG << "  Edges IDs: " << sizeof(EdgeID)
      << " bytes | Edge weights (Local): " << sizeof(LocalEdgeWeight)
      << " bytes | Edge weights (IP): " << sizeof(IPEdgeWeight) << " bytes";
}
} // namespace kaminpar::cio
