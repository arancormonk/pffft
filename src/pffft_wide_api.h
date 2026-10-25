/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Entry points of the 8-lane backend when it is its own translation unit
  (MSVC: pffft_wide_tu.c, compiled with /arch:AVX2, defines them; pffft.c
  calls them after the CPU check). Both include this file, so the compiler
  checks that the definitions match. Not used otherwise: then the backend
  is part of pffft.c and its functions are static.
*/

#ifndef PFFFT_WIDE_API_H
#define PFFFT_WIDE_API_H

PFFFT_Setup *pffft_w8_new_setup(int N, pffft_transform_t transform);
void pffft_w8_transform(const PFFFT_Setup *setup, const float *input, float *output, float *work,
                        pffft_direction_t direction);
void pffft_w8_transform_ordered(const PFFFT_Setup *setup, const float *input, float *output,
                                float *work, pffft_direction_t direction);
void pffft_w8_zreorder(const PFFFT_Setup *setup, const float *in, float *out, pffft_direction_t direction);
void pffft_w8_zconvolve_accumulate(const PFFFT_Setup *s, const float *a, const float *b, float *ab,
                                   float scaling);
void pffft_w8_zconvolve_scale(const PFFFT_Setup *s, const float *a, const float *b, float *ab,
                              float scaling);
void pffft_w8_zconvolve(const PFFFT_Setup *s, const float *a, const float *b, float *ab);
int pffft_w8_zconvert_zp(const PFFFT_Setup *s, const float *in, float *out, float scaling);
int pffft_w8_zconvolve_zp(const PFFFT_Setup *s, const float *x, const float *hzp, float *ab);

#endif /* PFFFT_WIDE_API_H */
