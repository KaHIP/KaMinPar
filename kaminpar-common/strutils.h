/*******************************************************************************
 * Utility functions for common string operations.
 *
 * @file:   strings.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace kaminpar::str {
std::string extract_basename(const std::string &path, bool keep_extension = false);
std::string to_lower(std::string arg);
std::vector<std::string> explode(const std::string &str, char del);
bool ends_with(const std::string &filename, const std::string &extension);

std::string &rtrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &ltrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &trim(std::string &s, const char *t = " \t\n\r\f\v");

template <typename Elements>
std::string implode(const Elements &elements, const std::string &separator) {
  if (elements.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << elements.front();
  for (std::size_t i = 1; i < elements.size(); ++i) {
    ss << separator << elements[i];
  }
  return ss.str();
}

std::string &replace_all(std::string &str, const std::string &replace, const std::string &with);

std::string &
replace_all(std::string &str, const std::vector<std::pair<std::string, std::string>> &replacements);
} // namespace kaminpar::str
