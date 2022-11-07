/*******************************************************************************
 * @file:   console_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#include "common/console_io.h"

#include <kassert/kassert.hpp>

#include "common/assertion_levels.h"
#include "common/logger.h"

namespace kaminpar::cio {
void print_delimiter(std::ostream& out, const char ch) {
    out << std::string(80, ch) << "\n";
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

void print_banner(const std::string& title) {
    LOG;
    LOG << std::string(80, '*');
    LOG << "* " << title << std::string(80 - 4 - title.size(), ' ') << " *";
    LOG << std::string(80, '*');
    LOG;
}
} // namespace kaminpar::cio
