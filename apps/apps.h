/*******************************************************************************
 * @file:   apps.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#if __has_include(<numa.h>)
    #include <numa.h>
#endif // __has_include(<numa.h>)

#include "common/logger.h"

namespace kaminpar {
inline void init_numa() {
#if __has_include(<numa.h>)
    if (numa_available() >= 0) {
        numa_set_interleave_mask(numa_all_nodes_ptr);
        return;
    }
#endif // __has_include(<numa.h>)
    LOG << "NUMA not available";
}
} // namespace kaminpar
