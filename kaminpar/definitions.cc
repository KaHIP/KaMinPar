/*******************************************************************************
 * @file:   definitions.cc
 *
 * @author: Daniel Seemaier
 * @date:   05.10.21
 * @brief:
 ******************************************************************************/
#include "definitions.h"

#ifdef KAMINPAR_BACKWARD_CPP
#include "backward.hpp"
#endif // KAMINPAR_BACKWARD_CPP

#include <sstream>

namespace kaminpar::debug {
void print_stacktrace() {
#ifdef KAMINPAR_BACKWARD_CPP
  using namespace backward;
  std::ostringstream oss;
  StackTrace st;
  st.load_here(10);
  Printer p;
  p.print(st, oss);

  std::cout << oss.str() << std::endl;
#endif // KAMINPAR_BACKWARD_CPP
}
} // namespace kaminpar::debug