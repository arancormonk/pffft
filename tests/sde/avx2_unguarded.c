/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/* Negative control for tests/run_sde_checks.sh: executes an AVX2 instruction
   without any CPU check, so Intel SDE must report it on a chip without AVX2.
   Built as an executable, or (AVX2_UNGUARDED_DLL) as a DLL plus a caller. */

#include <immintrin.h>
#include <stdio.h>

#if defined(_MSC_VER)
#  define NEG_EXPORT __declspec(dllexport)
#  define NEG_NOINLINE __declspec(noinline)
#else
#  define NEG_EXPORT
#  define NEG_NOINLINE __attribute__((noinline))
#endif

volatile int neg_buf[8];

#if defined(AVX2_UNGUARDED_LIB)
NEG_EXPORT NEG_NOINLINE int avx2_unguarded(int x) {
  /* through memory the compiler cannot see into, so that it cannot fold
     the vector arithmetic away */
  __m256i v;
  int k;
  for (k = 0; k < 8; ++k) neg_buf[k] = x + k;
  v = _mm256_loadu_si256((const __m256i *)(const void *)neg_buf);
  v = _mm256_add_epi32(v, v);              /* vpaddd ymm: AVX2 */
  _mm256_storeu_si256((__m256i *)(void *)neg_buf, v);
  return neg_buf[3];
}
#elif defined(AVX2_UNGUARDED_CALLER)
__declspec(dllimport) int avx2_unguarded(int x);
int main(int argc, char **argv) {
  (void)argv;
  printf("%d\n", avx2_unguarded(argc));
  return 0;
}
#else
static NEG_NOINLINE int avx2_unguarded(int x) {
  /* through memory the compiler cannot see into, so that it cannot fold
     the vector arithmetic away */
  __m256i v;
  int k;
  for (k = 0; k < 8; ++k) neg_buf[k] = x + k;
  v = _mm256_loadu_si256((const __m256i *)(const void *)neg_buf);
  v = _mm256_add_epi32(v, v);              /* vpaddd ymm: AVX2 */
  _mm256_storeu_si256((__m256i *)(void *)neg_buf, v);
  return neg_buf[3];
}
int main(int argc, char **argv) {
  (void)argv;
  printf("%d\n", avx2_unguarded(argc));
  return 0;
}
#endif
