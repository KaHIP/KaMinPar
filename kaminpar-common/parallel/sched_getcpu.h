/*******************************************************************************
 * @file:   sched_getcpu.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Dummy definition of sched_getcpu() for macOS.
 ******************************************************************************/
#pragma once

#if __has_include(<sched.h>)
#include <sched.h>
#define HAS_SCHED_GETCPU
#endif

#ifndef __linux__
inline int sched_getcpu() {
  return 0;
}
#endif
