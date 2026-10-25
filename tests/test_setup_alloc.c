/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 * Exercise both allocations inside each setup, including retry and cleanup.
 */
#include <assert.h>
#include <stddef.h>
#include <pffft/pffft.h>
#include <pffft/pffft_double.h>
#include "pffft_internal.h"

void *__real_malloc(size_t size);
void __real_free(void *ptr);
void *__wrap_malloc(size_t size);
void __wrap_free(void *ptr);
static int calls, fail_at, live;

void *__wrap_malloc(size_t size) {
  void *ptr;
  if (++calls == fail_at) return NULL;
  ptr = __real_malloc(size);
  if (ptr) ++live;
  return ptr;
}

void __wrap_free(void *ptr) {
  if (ptr) --live;
  __real_free(ptr);
}

int main(void) {
  int n, point;
  for (n = 256; n <= 512; n *= 2) {
    for (point = 1; point <= 2; ++point) {
#ifdef PFFFT_ENABLE_FLOAT
      PFFFT_Setup *s;
      calls = 0; fail_at = point;
      s = pffft_new_setup(n, PFFFT_REAL);
      assert(!s && calls == point && live == 0);
      calls = 0; fail_at = 0;
      s = pffft_new_setup(n, PFFFT_REAL);
      assert(s && calls == 2);
      pffft_destroy_setup(s);
      assert(live == 0);
      /* the 8-lane backend, where available: its failures are reported too,
         without falling back to (and allocating for) the 4-lane setup */
      calls = 0; fail_at = 0;
      s = pffft_internal_new_setup(n, PFFFT_REAL, PFFFT_BACKEND_W8);
      if (s) {
        assert(calls == 2 && pffft_setup_simd_size(s) == 8);
        pffft_destroy_setup(s);
        assert(live == 0);
        calls = 0; fail_at = point;
        s = pffft_internal_new_setup(n, PFFFT_REAL, PFFFT_BACKEND_W8);
        assert(!s && calls == point && live == 0);
      }
#endif
#ifdef PFFFT_ENABLE_DOUBLE
      {
        PFFFTD_Setup *sd;
        calls = 0; fail_at = point;
        sd = pffftd_new_setup(n, PFFFT_REAL);
        assert(!sd && calls == point && live == 0);
        calls = 0; fail_at = 0;
        sd = pffftd_new_setup(n, PFFFT_REAL);
        assert(sd && calls == 2);
        pffftd_destroy_setup(sd);
        assert(live == 0);
      }
#endif
    }
  }
  return 0;
}
