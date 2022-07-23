/*******************************************************************************
 * @file:   console_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#include "common/console_io.h"

#include <iostream>

namespace kaminpar::cio {
void print_banner(const std::string& title) {
    std::cout << std::endl;
    std::cout << std::string(80, '*') << std::endl;
    std::cout << "* " << title << std::string(80 - 4 - title.size(), ' ') << " *" << std::endl;
    std::cout << std::string(80, '*') << std::endl;
    std::cout << std::endl;
}
} // namespace kaminpar::cio
