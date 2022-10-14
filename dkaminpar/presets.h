/*******************************************************************************
 * @file:   presets.h
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Configuration presets.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"

namespace kaminpar::dist {
Context create_default_context();
Context create_strong_context();
} // namespace kaminpar::dist

