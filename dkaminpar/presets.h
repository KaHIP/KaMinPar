/*******************************************************************************
 * @file:   presets.h
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Configuration presets.
 ******************************************************************************/
#pragma once

// These functions are part of the public API
#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist {
Context create_tr_fast_context();
Context create_tr_strong_context();
Context create_jet_context();
} // namespace kaminpar::dist
