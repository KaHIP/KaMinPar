/*******************************************************************************
 * utils/counting_wait.h
 *
 * Simple counter that allows threads to wait until the counter is changed.
 * Waiting threads are sleeping using the futex syscall.
 *
 * Part of Project growt - https://github.com/TooBiased/growt.git
 *
 * Copyright (C) 2015-2016 Tobias Maier <t.maier@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#ifndef COUNTING_WAIT_H
#define COUNTING_WAIT_H


#include <atomic>
#include <iostream>
#include <memory>
#include <cstdlib>

#if defined(__linux__)
#include <sys/time.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif 

namespace growt
{

#if defined(__linux__)
static long sys_futex(void* addr1, int op, int val1, struct timespec* timeout,
                      void* addr2, int val3)
{
    return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}
#endif 

class alignas(64) counting_wait
{
  public:
    inline counting_wait(int start = 0) : counter(start)
    {
        if (sizeof(std::atomic_int) != 4)
            std::cout << "std::atomic_int has wrong size:"
                      << sizeof(std::atomic_int) << std::endl;
    }

    inline bool inc_if(int exp)
    {
        auto temp = exp;
        return counter.compare_exchange_strong(temp, exp + 1,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire);
    }

    inline bool wait_if(int exp)
    {
#if defined(__linux__)
        // while (counter.load(std::memory_order_acquire) < l_epoch) ;
        // //temporary should soon be removed
        auto ecode = sys_futex(&counter, FUTEX_WAIT, exp, NULL, NULL, 0);
        return !ecode;
#else 
        counter.wait(exp);
        return true; // always ignored
#endif 
    }

    inline unsigned int wake(int n_threads = 9999) // always 9999
    {
#if defined(__linux__)
        return sys_futex(&counter, FUTEX_WAKE, n_threads, NULL, NULL, 0);
#else 
        counter.notify_all();
        return 1; // always ignored
#endif 
    }

  private:
    std::atomic_int counter;
};

} // namespace growt

#endif // COUNTING_WAIT_H
