/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Layout of the setup structure, shared by pffft_priv_impl.h (PFFFT_Setup,
  PFFFTD_Setup) and the 8-lane backend when MSVC compiles that as its own
  translation unit (pffft_wide_tu.c), so that both sides agree on it.
*/

#ifndef PFFFT_SETUP_H
#define PFFFT_SETUP_H

/* largest accepted N: larger values overflow int in the setup and passes */
#define PFFFT_MAX_N (1<<26)

#ifndef IFAC_MAX_SIZE
#define IFAC_MAX_SIZE 25 /* max number of integer factors for the decomposition, +2 */
#endif

#define PFFFT_SETUP_KIND_NATIVE 0  /* the SIMD_SZ-lane implementation of pffft_priv_impl.h */
#define PFFFT_SETUP_KIND_W8     1  /* 8-lane AVX2 float backend, pffft_wide_impl.h */

/*
  N
  Ncvec      nb of complex simd vectors (N/4 if PFFFT_COMPLEX, N/8 if PFFFT_REAL)
  ifac       the decomposition of N into small integers: N, number of
             factors, factors (admitted values: 2, 3, 4, 5 or 8)
  kind       PFFFT_SETUP_KIND_*: which backend created the setup
  data       allocated room for twiddle coefs
  e          points into 'data', N/4*3 elements
  twiddle    points into 'data', N/4 elements
*/
#define PFFFT_SETUP_FIELDS(vector_type, real_type) \
  int N;                                           \
  int Ncvec;                                       \
  int ifac[IFAC_MAX_SIZE];                         \
  pffft_transform_t transform;                     \
  int kind;                                        \
  vector_type *data;                               \
  real_type *e;                                    \
  real_type *twiddle;

#endif /* PFFFT_SETUP_H */
