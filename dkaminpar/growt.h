/*******************************************************************************
 * @file:   growt.h
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:  Include growt and suppress -Wpedantic warnings.
 ******************************************************************************/
#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <tbb/enumerable_thread_specific.h>
#include <utils/hash/murmur2_hash.hpp>
#pragma GCC diagnostic pop