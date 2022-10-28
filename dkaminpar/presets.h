/*******************************************************************************
 * @file:   presets.h
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Configuration presets.
 ******************************************************************************/
#pragma once

#include <string>
#include <unordered_set>

#include "dkaminpar/context.h"

namespace kaminpar::dist {
Context                         create_context_by_preset_name(const std::string& name);
std::unordered_set<std::string> get_preset_names();

Context create_default_context();
Context create_strong_context();

// Configurations used in the IPDPS'23 submission
Context create_ipdps23_submission_default_context();
Context create_ipdps23_submission_strong_context();
} // namespace kaminpar::dist
