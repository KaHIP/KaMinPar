/*******************************************************************************
 * @file:   enum_string_conversion.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper macros to convert between strings and enum elements.
 ******************************************************************************/
#pragma once

#define DECLARE_ENUM_STRING_CONVERSION(type_name, prefix_name)            \
    type_name     prefix_name##_from_string(const std::string& searched); \
    std::ostream& operator<<(std::ostream& os, const type_name& value);   \
    std::string   prefix_name##_names(const std::string& sep = ", ")

#define DEFINE_ENUM_STRING_CONVERSION(type_name, prefix_name)            \
    struct type_name##Dummy {                                            \
        static std::map<type_name, std::string_view> enum_to_name;       \
    };                                                                   \
                                                                         \
    type_name prefix_name##_from_string(const std::string& searched) {   \
        for (const auto [value, name]: type_name##Dummy::enum_to_name) { \
            if (name == searched) {                                      \
                return value;                                            \
            }                                                            \
        }                                                                \
        throw std::runtime_error("invalid name: "s + searched);          \
    }                                                                    \
                                                                         \
    std::ostream& operator<<(std::ostream& os, const type_name& value) { \
        return os << type_name##Dummy::enum_to_name.find(value)->second; \
    }                                                                    \
                                                                         \
    std::string prefix_name##_names(const std::string& sep) {            \
        std::stringstream names;                                         \
        bool              first = true;                                  \
        for (const auto [value, name]: type_name##Dummy::enum_to_name) { \
            if (!first) {                                                \
                names << sep;                                            \
            }                                                            \
            names << name;                                               \
            first = false;                                               \
        }                                                                \
        return names.str();                                              \
    }                                                                    \
                                                                         \
    std::map<type_name, std::string_view> type_name##Dummy::enum_to_name
