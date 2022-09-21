/*******************************************************************************
 * @file:   console_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#include "common/console_io.h"

#include <kassert/kassert.hpp>

#include "common/assert.h"
#include "common/detect_macro.h"
#include "common/logger.h"

namespace kaminpar::cio {
void print_delimiter() {
    LOG << "################################################################################";
}
void print_kaminpar_banner() {
    print_delimiter();
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    LOG << "#                _  __       __  __  _         ____                 #ASSERTIONS#";
    LOG << "#               | |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __    ############";
#else
    LOG << "#                _  __       __  __  _         ____                            #";
    LOG << "#               | |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __               #";
#endif
    LOG << "#               | ' / / _` || |\\/| || || '_ \\ | |_) |/ _` || '__|              #";
    LOG << "#               | . \\| (_| || |  | || || | | ||  __/| (_| || |                 #";
    LOG << "#               |_|\\_\\\\__,_||_|  |_||_||_| |_||_|    \\__,_||_|                 #";
    LOG << "#                                                                              #";
    print_delimiter();
}

void print_dkaminpar_banner() {
    print_delimiter();
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    LOG << "#                _  _  __       __  __  _         ____              #ASSERTIONS#";
    LOG << "#             __| || |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __ ############";
#else
    LOG << "#                _  _  __       __  __  _         ____                         #";
    LOG << "#             __| || |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __            #";
#endif
    LOG << "#            / _` || ' / / _` || |\\/| || || '_ \\ | |_) |/ _` || '__|           #";
    LOG << "#           | (_| || . \\| (_| || |  | || || | | ||  __/| (_| || |              #";
    LOG << "#            \\__,_||_|\\_\\\\__,_||_|  |_||_||_| |_||_|    \\__,_||_|              #";
    LOG << "#                                                                              #";
    print_delimiter();
}

template <typename NodeID, typename EdgeID, typename NodeWeight, typename EdgeWeight>
void print_build_identifier(const std::string& commit, const std::string& hostname) {
    LOG << "Current commit hash:          " << (commit.empty() ? "<not available>" : commit);
    std::string assertion_level_name = "always";
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_LIGHT) {
        assertion_level_name += "+light";
    }
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_NORMAL) {
        assertion_level_name += "+normal";
    }
    if (KASSERT_ASSERTION_LEVEL >= ASSERTION_LEVEL_HEAVY) {
        assertion_level_name += "+heavy";
    }
    LOG << "Assertion level:              " << assertion_level_name;
    LOG << "Statistics:                   " << (DETECT_EXIST(KAMINPAR_ENABLE_STATISTICS) ? "enabled" : "disabled");
    LOG << "Data type sizes:";
    LOG << "  Nodes:        " << sizeof(NodeID) << " bytes | Edges:        " << sizeof(EdgeID) << " bytes";
    LOG << "  Node weights: " << sizeof(NodeWeight) << " bytes | Edge weights: " << sizeof(EdgeWeight) << " bytes";
    LOG << "Built on:                     " << (hostname.empty() ? "<not available>" : hostname);
    LOG << "################################################################################";
}

void print_banner(const std::string& title) {
    LOG;
    LOG << std::string(80, '*');
    LOG << "* " << title << std::string(80 - 4 - title.size(), ' ') << " *";
    LOG << std::string(80, '*');
    LOG;
}
} // namespace kaminpar::cio
