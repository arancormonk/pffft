/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  MSVC: the 8-lane AVX2 float backend as its own translation unit. Compile
  this file with /arch:AVX2 and define PFFFT_WIDE_TU for it and for pffft.c
  (CMakeLists.txt does both for MSVC x64 builds). pffft.c then runs the
  backend only after its CPU check. With other compilers, or without
  PFFFT_WIDE_TU, this file is empty: GCC and Clang build the backend inside
  pffft.c.
*/

#if defined(PFFFT_WIDE_TU) && defined(_MSC_VER) && !defined(__clang__)
#  if !defined(__AVX2__)
#    error "pffft_wide_tu.c must be compiled with /arch:AVX2"
#  endif
#  define PFFFT_WIDE_TU_PART 1
#  include "pffft.c"
#else
typedef int pffft_wide_tu_unused;  /* ISO C: a translation unit is not empty */
#endif
