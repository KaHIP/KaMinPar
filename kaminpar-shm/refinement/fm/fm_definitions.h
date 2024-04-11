/*******************************************************************************
 * Basic definitions used by FM refinement code.
 *
 * @file:   fm_definitions.h
 * @author: Daniel Seemaier
 * @date:   27.02.2024
 ******************************************************************************/
#pragma once 

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::fm {
struct Move {
  NodeID node;
  BlockID from;
};

struct AppliedMove {
  NodeID node;
  BlockID from;
  bool improvement;
};
}
