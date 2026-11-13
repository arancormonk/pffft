/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Tests of src/pffft_pass_config.h: what it leaves in PFFFT_TERMINAL_RADIX8
  and PFFFT_INPLACE_FIRST_PASS for the defaults and for definitions made
  before it, among them expressions without parentheses such as 0 ? 0 : 1.

  The header is included once per case, each time with the definitions of
  that case (the include guard, the two switches and SIMD_SZ undefined in
  between). Right after each inclusion the switches are captured in a
  t_case: their expansion as text, which must be the literal 0 or 1, and
  expressions whose value depends on how the expansion groups: 2 * X (2
  for the literal 1, 1 for 0 ? 0 : 1 and for 1 - 1), X - 1 (-1 for the
  literal 0, 0 for 0 && 1), -X (-1 for the literal 1, 1 for 0 ? 0 : 1),
  X && 0 (0 for the literal, 1 for 1 || 0), and whether a block
  #if X && 0 was compiled. The definitions of the build, if any, are
  undefined before the first case.

  Needs no library. Prints one line per failing check and exits with a
  non-zero status if any check failed.
*/

#include <stdio.h>
#include <string.h>

#define T_STR_(x) #x
#define T_STR(x) T_STR_(x)

/* the switches after one inclusion of the header, and their values */
typedef struct {
  const char *what;               /* the case */
  int r8, ip;                     /* the values the switches must have */
  const char *r8_text, *ip_text;  /* their expansions */
  int r8_twice, ip_twice;         /* 2 * X */
  int r8_less1, ip_less1;         /* X - 1 */
  int r8_neg, ip_neg;             /* -X */
  int r8_and0, ip_and0;           /* X && 0 */
  int r8_if_and0, ip_if_and0;     /* 1 if a block #if X && 0 was compiled */
} t_case;

/* a t_case of the switches as the header left them; r8 and ip are the
   values they must have. T_IF_R8_AND0 and T_IF_IP_AND0 are set by the
   block of the case. */
#define T_CAPTURE(what, r8, ip) {                                     \
    what, r8, ip,                                                     \
    T_STR(PFFFT_TERMINAL_RADIX8), T_STR(PFFFT_INPLACE_FIRST_PASS),    \
    2 * PFFFT_TERMINAL_RADIX8, 2 * PFFFT_INPLACE_FIRST_PASS,          \
    PFFFT_TERMINAL_RADIX8 - 1, PFFFT_INPLACE_FIRST_PASS - 1,          \
    -PFFFT_TERMINAL_RADIX8, -PFFFT_INPLACE_FIRST_PASS,                \
    PFFFT_TERMINAL_RADIX8 && 0, PFFFT_INPLACE_FIRST_PASS && 0,        \
    T_IF_R8_AND0, T_IF_IP_AND0 }

/* 1: no definition, 4 lanes: both 1 */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#define SIMD_SZ 4
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_1 = T_CAPTURE("defaults, SIMD_SZ 4", 1, 1);

/* 2: no definition, single lane: the terminal radix-8 pass is off */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_2 = T_CAPTURE("defaults, SIMD_SZ 1", 0, 1);

/* 3: expressions of value 1 without parentheses */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 4
#define PFFFT_TERMINAL_RADIX8 0 ? 0 : 1
#define PFFFT_INPLACE_FIRST_PASS 1 || 0
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_3 = T_CAPTURE("0 ? 0 : 1 and 1 || 0, SIMD_SZ 4", 1, 1);

/* 4: the same, the other way round, single lane (a definition overrides
   the default of SIMD_SZ 1) */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 1
#define PFFFT_TERMINAL_RADIX8 1 || 0
#define PFFFT_INPLACE_FIRST_PASS 0 ? 0 : 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_4 = T_CAPTURE("1 || 0 and 0 ? 0 : 1, SIMD_SZ 1", 1, 1);

/* 5: expressions of value 0 without parentheses */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 4
#define PFFFT_TERMINAL_RADIX8 1 - 1
#define PFFFT_INPLACE_FIRST_PASS 0 && 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_5 = T_CAPTURE("1 - 1 and 0 && 1, SIMD_SZ 4", 0, 0);

/* 6: the same, the other way round */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 4
#define PFFFT_TERMINAL_RADIX8 0 && 1
#define PFFFT_INPLACE_FIRST_PASS 1 - 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_6 = T_CAPTURE("0 && 1 and 1 - 1, SIMD_SZ 4", 0, 0);

/* 7: parenthesised */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 4
#define PFFFT_TERMINAL_RADIX8 (1)
#define PFFFT_INPLACE_FIRST_PASS (0)
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_7 = T_CAPTURE("(1) and (0), SIMD_SZ 4", 1, 0);

/* 8: parenthesised, the other way round, single lane */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 1
#define PFFFT_TERMINAL_RADIX8 (0)
#define PFFFT_INPLACE_FIRST_PASS (1)
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_8 = T_CAPTURE("(0) and (1), SIMD_SZ 1", 0, 1);

/* 9: plain literals (as CMake defines them for ON and OFF) */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 4
#define PFFFT_TERMINAL_RADIX8 1
#define PFFFT_INPLACE_FIRST_PASS 0
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_9 = T_CAPTURE("1 and 0, SIMD_SZ 4", 1, 0);

/* 10: plain literals, the other way round, single lane */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 1
#define PFFFT_TERMINAL_RADIX8 0
#define PFFFT_INPLACE_FIRST_PASS 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#  define T_IF_R8_AND0 1
#else
#  define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#  define T_IF_IP_AND0 1
#else
#  define T_IF_IP_AND0 0
#endif
static const t_case t_case_10 = T_CAPTURE("0 and 1, SIMD_SZ 1", 0, 1);

static const t_case *const t_cases[] = {
  &t_case_1, &t_case_2, &t_case_3, &t_case_4, &t_case_5,
  &t_case_6, &t_case_7, &t_case_8, &t_case_9, &t_case_10
};

/* 1 and a report line when got != expected, else 0 */
static int t_int(const char *what, const char *check, int got, int expected) {
  if (got == expected) return 0;
  printf("FAIL %s: %s is %d, expected %d\n", what, check, got, expected);
  return 1;
}

/* 1 and a report line when the expansion is not the literal of value,
   else 0 */
static int t_text(const char *what, const char *check, const char *got, int value) {
  const char *expected = value ? "1" : "0";
  if (strcmp(got, expected) == 0) return 0;
  printf("FAIL %s: %s expands to \"%s\", expected \"%s\"\n", what, check, got, expected);
  return 1;
}

static int t_check_case(const t_case *c) {
  int fails = 0;
  fails += t_text(c->what, "PFFFT_TERMINAL_RADIX8", c->r8_text, c->r8);
  fails += t_text(c->what, "PFFFT_INPLACE_FIRST_PASS", c->ip_text, c->ip);
  fails += t_int(c->what, "2 * PFFFT_TERMINAL_RADIX8", c->r8_twice, 2 * c->r8);
  fails += t_int(c->what, "2 * PFFFT_INPLACE_FIRST_PASS", c->ip_twice, 2 * c->ip);
  fails += t_int(c->what, "PFFFT_TERMINAL_RADIX8 - 1", c->r8_less1, c->r8 - 1);
  fails += t_int(c->what, "PFFFT_INPLACE_FIRST_PASS - 1", c->ip_less1, c->ip - 1);
  fails += t_int(c->what, "-PFFFT_TERMINAL_RADIX8", c->r8_neg, -c->r8);
  fails += t_int(c->what, "-PFFFT_INPLACE_FIRST_PASS", c->ip_neg, -c->ip);
  fails += t_int(c->what, "PFFFT_TERMINAL_RADIX8 && 0", c->r8_and0, 0);
  fails += t_int(c->what, "PFFFT_INPLACE_FIRST_PASS && 0", c->ip_and0, 0);
  fails += t_int(c->what, "#if PFFFT_TERMINAL_RADIX8 && 0 compiled", c->r8_if_and0, 0);
  fails += t_int(c->what, "#if PFFFT_INPLACE_FIRST_PASS && 0 compiled", c->ip_if_and0, 0);
  return fails;
}

int main(void) {
  int fails = 0;
  size_t i;
  for (i = 0; i < sizeof(t_cases) / sizeof(t_cases[0]); ++i)
    fails += t_check_case(t_cases[i]);
  if (fails) {
    printf("%d check(s) failed\n", fails);
    return 1;
  }
  printf("all checks passed (%d cases)\n", (int)(sizeof(t_cases) / sizeof(t_cases[0])));
  return 0;
}
