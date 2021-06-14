#pragma once

#include "definitions.h"
#include "utility/math.h"

#include <algorithm>
#include <cctype>
#include <concepts>

namespace kaminpar::utility {
namespace str {
std::string extract_basename(const std::string &path);
std::string to_lower(std::string arg);
std::vector<std::string> explode(const std::string &str, char del);

std::string &rtrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &ltrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &trim(std::string &s, const char *t = " \t\n\r\f\v");
} // namespace str
} // namespace kaminpar::utility