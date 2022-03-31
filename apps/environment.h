#pragma once

#include <string>

namespace kaminpar {
struct Environment {
    static const std::string GIT_SHA1;
    static const std::string GIT_MODIFIED_FILES;
    static const std::string HOSTNAME;
};
} // namespace kaminpar