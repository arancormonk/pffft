/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Library-internal entry points of pffft.c, for the tests. Not installed and
  not part of the stable API.
*/

#ifndef PFFFT_INTERNAL_H
#define PFFFT_INTERNAL_H

/* include pffft.h first */

#ifdef __cplusplus
extern "C" {
#endif

#define PFFFT_BACKEND_AUTO   0  /* what pffft_new_setup() picks */
#define PFFFT_BACKEND_NATIVE 1  /* the pffft_simd_size()-lane implementation */
#define PFFFT_BACKEND_W8     2  /* the 8-lane AVX2 backend; NULL when unavailable */

/*
  pffft_new_setup() with a forced backend. PFFFT_BACKEND_W8 ignores the
  CPU-model and size tables (but not the instruction-set checks) so tests
  can run the 8-lane backend on any CPU that supports it. Returns NULL
  where pffft_new_setup() would, or when the requested backend is not
  built in, not supported by the CPU, or does not support N.
*/
PFFFT_EXPORT PFFFT_Setup *pffft_internal_new_setup(int N, pffft_transform_t transform, int backend);

#ifdef __cplusplus
}
#endif

#endif /* PFFFT_INTERNAL_H */
