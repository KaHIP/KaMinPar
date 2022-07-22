/*******************************************************************************
 * @file:   console_io.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#pragma once

#include <mutex>

namespace kaminpar::cio {
void print_banner(const std::string& title);
} // namespace kaminpar::cio
