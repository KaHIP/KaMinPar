/*******************************************************************************
 * @file:   definitions.cc
 *
 * @author: Daniel Seemaier
 * @date:   05.10.21
 * @brief:
 ******************************************************************************/
#include "definitions.h"

#ifdef USE_BACKWARD
#include "backward.hpp"
#endif // USE_BACKWARD

#include <sstream>

namespace kaminpar::debug {
void print_stacktrace() {
#ifdef USE_BACKWARD
  using namespace backward;
  std::ostringstream oss;
  StackTrace st;
  st.load_here(32);
  Printer p;
  p.print(st, oss);

  std::cout << oss.str() << std::endl;
#endif // USE_BACKWARD
}
} // namespace kaminpar::debug