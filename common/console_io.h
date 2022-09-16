/*******************************************************************************
 * @file:   console_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#pragma once

#include <string>

namespace kaminpar::cio {
void print_kaminpar_banner();
void print_dkaminpar_banner();
void print_build_identifier(const std::string& commit, const std::string& hostname);
void print_banner(const std::string& title);
} // namespace kaminpar::cio
