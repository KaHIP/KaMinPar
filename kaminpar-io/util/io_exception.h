/*******************************************************************************
 * Common exception type for IO errors.
 *
 * @file:   io_exception.h
 ******************************************************************************/
#pragma once

#include <stdexcept>
#include <string>

namespace kaminpar::io {

class IOException : public std::runtime_error {
public:
  explicit IOException(const std::string &msg) : std::runtime_error(msg) {}
  explicit IOException(const char *msg) : std::runtime_error(msg) {}
};

} // namespace kaminpar::io
