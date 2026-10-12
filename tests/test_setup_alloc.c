/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Mark Connor
 * Exercise both allocations inside each setup, including retry and cleanup.
 */
#include <assert.h>
#include <stddef.h>
#include <pffft/pffft.h>
#include <pffft/pffft_double.h>

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
#endif
#ifdef PFFFT_ENABLE_DOUBLE
      {
        PFFFTD_Setup *s;
        calls = 0; fail_at = point;
        s = pffftd_new_setup(n, PFFFT_REAL);
        assert(!s && calls == point && live == 0);
        calls = 0; fail_at = 0;
        s = pffftd_new_setup(n, PFFFT_REAL);
        assert(s && calls == 2);
        pffftd_destroy_setup(s);
        assert(live == 0);
      }
#endif
    }
  }
  return 0;
}
