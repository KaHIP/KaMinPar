#pragma once

#include <string>
#include <unordered_set>

#include "kaminpar/context.h"

namespace kaminpar::shm {
Context create_context_by_preset_name(const std::string &name);
std::unordered_set<std::string> get_preset_names();

Context create_default_context();
Context create_largek_context();
} // namespace kaminpar::shm
