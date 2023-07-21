/*******************************************************************************
 * Configuration presets.
 *
 * @file:   presets.h
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 ******************************************************************************/
#pragma once

// These functions are part of the public API
#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist {
Context create_europar23_fast_context();
Context create_europar23_strong_context();
} // namespace kaminpar::dist
