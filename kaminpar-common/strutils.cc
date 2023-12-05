/*******************************************************************************
 * Utility functions for common string operations.
 *
 * @file:   strings.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/strutils.h"

#include <sstream>

namespace kaminpar::str {
std::string extract_basename(const std::string &path, const bool keep_extension) {
  const std::size_t slash = path.find_last_of('/');
  const std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
  return keep_extension ? name : name.substr(0, name.find_last_of('.'));
}

std::string to_lower(std::string arg) {
  std::transform(arg.begin(), arg.end(), arg.begin(), [](const char c) { return std::tolower(c); });
  return arg;
}

bool ends_with(const std::string &filename, const std::string &extension) {
  return extension.length() <= filename.length() &&
         std::equal(extension.rbegin(), extension.rend(), filename.rbegin());
}

std::vector<std::string> explode(const std::string &str, const char del) {
  std::stringstream ss(str);
  std::vector<std::string> splits;
  std::string part;
  while (std::getline(ss, part, del)) {
    splits.push_back(part);
  }
  return splits;
}

std::string &rtrim(std::string &s, const char *t) {
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

std::string &ltrim(std::string &s, const char *t) {
  s.erase(0, s.find_first_not_of(t));
  return s;
}

std::string &trim(std::string &s, const char *t) {
  return ltrim(rtrim(s, t), t);
}

std::string &replace_all(std::string &str, const std::string &replace, const std::string &with) {
  for (auto pos = str.find(replace); pos != std::string::npos;
       pos = str.find(replace, pos + with.length())) {
    str.replace(pos, replace.length(), with);
  }
  return str;
}

std::string &
replace_all(std::string &str, const std::vector<std::pair<std::string, std::string>> &replacements) {
  for (auto &replacement : replacements) {
    replace_all(str, replacement.first, replacement.second);
  }
  return str;
}
} // namespace kaminpar::str
