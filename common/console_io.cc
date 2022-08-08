/*******************************************************************************
 * @file:   console_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#include "common/console_io.h"

#include "common/logger.h"

namespace kaminpar::cio {
void print_banner(const std::string& title) {
    LOG;
    LOG << std::string(80, '*');
    LOG << "* " << title << std::string(80 - 4 - title.size(), ' ') << " *";
    LOG << std::string(80, '*');
    LOG;
}
} // namespace kaminpar::cio
