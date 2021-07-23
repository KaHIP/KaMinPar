/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#define DECLARE_ENUM_STRING_CONVERSION(type_name, prefix_name)                                                         \
  type_name prefix_name##_from_string(const std::string &searched);                                                    \
  std::ostream &operator<<(std::ostream &os, const type_name &value);                                                  \
  std::string prefix_name##_names(const std::string &sep = ", ")

#define DEFINE_ENUM_STRING_CONVERSION(type_name, prefix_name)                                                          \
  struct type_name##Dummy {                                                                                            \
    static std::map<type_name, std::string_view> enum_to_name;                                                         \
  };                                                                                                                   \
                                                                                                                       \
  type_name prefix_name##_from_string(const std::string &searched) {                                                   \
    for (const auto [value, name] : type_name##Dummy::enum_to_name) {                                                  \
      if (name == searched) { return value; }                                                                          \
    }                                                                                                                  \
    throw std::runtime_error("invalid name: "s + searched);                                                            \
  }                                                                                                                    \
                                                                                                                       \
  std::ostream &operator<<(std::ostream &os, const type_name &value) {                                                 \
    return os << type_name##Dummy::enum_to_name.find(value)->second;                                                   \
  }                                                                                                                    \
                                                                                                                       \
  std::string prefix_name##_names(const std::string &sep) {                                                            \
    std::stringstream names;                                                                                           \
    bool first = true;                                                                                                 \
    for (const auto [value, name] : type_name##Dummy::enum_to_name) {                                                  \
      if (!first) { names << sep; }                                                                                    \
      names << name;                                                                                                   \
      first = false;                                                                                                   \
    }                                                                                                                  \
    return names.str();                                                                                                \
  }                                                                                                                    \
                                                                                                                       \
  std::map<type_name, std::string_view> type_name##Dummy::enum_to_name
