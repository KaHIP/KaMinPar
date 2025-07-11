/*******************************************************************************
 * Helper functions for console IO.
 *
 * @file:   console_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-common/logger.h"

namespace kaminpar::cio {

void print_delimiter(const std::string &caption = "", char ch = '#');
void print_kaminpar_banner();
void print_dkaminpar_banner();
void print_build_identifier();

template <typename NodeID, typename EdgeID, typename NodeWeight, typename EdgeWeight>
void print_build_datatypes() {
  LOG << "Types:                        NID: " << sizeof(NodeID) << ", EID: " << sizeof(EdgeID)
      << ", NW: " << sizeof(NodeWeight) << ", EW: " << sizeof(EdgeWeight);
}

template <
    typename NodeID,
    typename EdgeID,
    typename LocalNodeWeight,
    typename LocalEdgeWeight,
    typename IPNodeWeight,
    typename IPEdgeWeight>
void print_build_datatypes() {
  LOG << "Types:                        NID: " << sizeof(NodeID) << ", EID: " << sizeof(EdgeID)
      << ", NW: " << sizeof(LocalNodeWeight) << "/" << sizeof(IPNodeWeight)
      << ", EW: " << sizeof(LocalEdgeWeight) << "/" << sizeof(IPEdgeWeight);
}

} // namespace kaminpar::cio
