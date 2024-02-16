/*******************************************************************************
 * Assertion levels to be used with KASSERT().
 *
 * @file:   assert.h
 * @author: Daniel Seemaier
 * @date:   14.06.2022
 ******************************************************************************/
#pragma once

#include <kassert/kassert.hpp>

namespace kaminpar::assert {
#define ASSERTION_LEVEL_ALWAYS 0
constexpr int always = ASSERTION_LEVEL_ALWAYS;
#define ASSERTION_LEVEL_LIGHT 10
constexpr int light = ASSERTION_LEVEL_LIGHT;
#define ASSERTION_LEVEL_NORMAL 30
constexpr int normal = ASSERTION_LEVEL_NORMAL; // same value as defined in KASSERT
#define ASSERTION_LEVEL_HEAVY 40
constexpr int heavy = ASSERTION_LEVEL_HEAVY;
} // namespace kaminpar::assert
