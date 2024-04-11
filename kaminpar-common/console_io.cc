/*******************************************************************************
 * Helper functions for console IO.
 *
 * @file:   console_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/console_io.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::cio {
void print_delimiter(const std::string &caption, const char ch) {
  if (caption.empty()) {
    LOG << std::string(80, ch);
  } else {
    LOG << std::string(80 - caption.size() - 5, ch) << " " << caption << " " << std::string(3, ch);
  }
}

void print_kaminpar_banner() {
  print_delimiter();
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  LOG << "#                _  __       __  __  _         ____                 "
         "#ASSERTIONS#";
  LOG << "#               | |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __   "
         " ############";
#else
  LOG << "#                _  __       __  __  _         ____                  "
         "          #";
  LOG << "#               | |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ __   "
         "            #";
#endif
  LOG << "#               | ' / / _` || |\\/| || || '_ \\ | |_) |/ _` || '__|  "
         "            #";
  LOG << "#               | . \\| (_| || |  | || || | | ||  __/| (_| || |      "
         "           #";
  LOG << "#               |_|\\_\\\\__,_||_|  |_||_||_| |_||_|    \\__,_||_|   "
         "              #";
  LOG << "#                                                                    "
         "          #";
  print_delimiter();
}

void print_dkaminpar_banner() {
  print_delimiter();
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  LOG << "#                _  _  __       __  __  _         ____              "
         "#ASSERTIONS#";
  LOG << "#             __| || |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ "
         "__ ############";
#else
  LOG << "#                _  _  __       __  __  _         ____               "
         "          #";
  LOG << "#             __| || |/ / __ _ |  \\/  |(_) _ __  |  _ \\  __ _  _ "
         "__            #";
#endif
  LOG << "#            / _` || ' / / _` || |\\/| || || '_ \\ | |_) |/ _` || "
         "'__|           #";
  LOG << "#           | (_| || . \\| (_| || |  | || || | | ||  __/| (_| || |   "
         "           #";
  LOG << "#            \\__,_||_|\\_\\\\__,_||_|  |_||_||_| |_||_|    "
         "\\__,_||_|              #";
  LOG << "#                                                                    "
         "          #";
  print_delimiter();
}

void print_build_identifier() {
  LOG << "Current commit hash:          "
      << (Environment::GIT_SHA1.empty() ? "<not available>" : Environment::GIT_SHA1);
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
#ifdef KAMINPAR_ENABLE_STATISTICS
  LOG << "Statistics:                   enabled";
#else  // KAMINPAR_ENABLE_STATISTICS
  LOG << "Statistics:                   disabled";
#endif // KAMINPAR_ENABLE_STATISTICS
  LOG << "Built on:                     "
      << (Environment::HOSTNAME.empty() ? "<not available>" : Environment::HOSTNAME);
}
} // namespace kaminpar::cio
