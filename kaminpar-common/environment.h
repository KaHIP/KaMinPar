/*******************************************************************************
 * Utility struct to pass in the current Git version and build system from
 * CMake.
 *
 * @file:   environment.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <string_view>

namespace kaminpar {

struct Environment {
  static const std::string_view GIT_SHA1;
  static const std::string_view GIT_MODIFIED_FILES;
  static const std::string_view HOSTNAME;
};

} // namespace kaminpar
