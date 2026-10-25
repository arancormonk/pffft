/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Contract tests for the setup-selected 8-lane AVX2 float backend
  (src/pffft_wide_impl.h); test_pffft_accuracy checks its numerics.

  - CPU detection: decoding truth table, and the order of the probes
    (XGETBV only after CPUID reports OSXSAVE and AVX, CPUID leaf 7 only
    when leaf 0 reports it), through fake CPUID/XGETBV operations.
  - Backend selection: deterministic, sizes the backend does not handle
    stay with the 4-lane implementation, and the default picks it only at
    the measured sizes of pffft_wide_sizes.h.
  - Alignment: the API promises only 16-byte aligned buffers. All eight
    transform modes and every spectral operation run with each distinct
    buffer (input, second operand, output, work) at 16 mod 32 bytes or 64-byte
    aligned, independently, with every alias pattern the API allows and
    work = NULL, and must give bit-identical results to separate 64-byte
    aligned buffers without touching guard regions around them.
  - Concurrency: setups are created and used from several threads.
*/

#include "pffft/pffft.h"
#include "pffft_internal.h"
#include "pffft_cpu_x86.h"
#include "pffft_wide_sizes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef PFFFT_TEST_THREADS
#include <pthread.h>
#endif

static int failures = 0;

static void check(int ok, const char *what, int a, int b) {
  if (!ok) {
    ++failures;
    printf("FAIL: %s (%d, %d)\n", what, a, b);
  }
}

/* ---------------- CPU detection ---------------- */

#define ECX_ALL (PFFFT_X86_ECX1_OSXSAVE | PFFFT_X86_ECX1_AVX | PFFFT_X86_ECX1_FMA)

static void test_decode(void) {
  const unsigned avx2 = PFFFT_X86_EBX7_AVX2;
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, 0x7, 1) == 1, "decode: all features", 0, 0);
  check(pffft_x86_decode_wide(0xd, ECX_ALL, avx2, 0xe7, 1) == 1, "decode: AVX-512 state too", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, 0x7, 1) == 0, "decode: no FMA", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, 0x7, 0) == 1, "decode: no FMA, not needed", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_AVX, avx2, 0x7, 1) == 0, "decode: no AVX", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_OSXSAVE, avx2, 0x7, 1) == 0, "decode: no OSXSAVE", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, 0, 0x7, 1) == 0, "decode: no AVX2", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, 0x5, 1) == 0, "decode: no XMM state", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, 0x3, 1) == 0, "decode: no YMM state", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, 0x1, 1) == 0, "decode: XCR0 = 1", 0, 0);
  check(pffft_x86_decode_wide(6, ECX_ALL, avx2, 0x7, 1) == 0, "decode: max leaf 6", 0, 0);
}

typedef struct {
  unsigned max_leaf, vendor[3], eax1, ecx1, ebx7;
  unsigned long long xcr0;
  int xgetbv_calls, leaf1_calls, leaf7_calls, bad;
} fake_cpu;

static void fake_cpuid(void *ctx, unsigned leaf, unsigned subleaf, unsigned *r) {
  fake_cpu *f = (fake_cpu*)ctx;
  r[0] = r[1] = r[2] = r[3] = 0;
  if (leaf > f->max_leaf) f->bad = 1;   /* unsupported leaf queried */
  if (leaf == 0) {
    r[0] = f->max_leaf; r[1] = f->vendor[0]; r[3] = f->vendor[1]; r[2] = f->vendor[2];
  } else if (leaf == 1) {
    ++f->leaf1_calls;
    r[0] = f->eax1; r[2] = f->ecx1;
  } else if (leaf == 7) {
    ++f->leaf7_calls;
    if (subleaf == 0) r[1] = f->ebx7;
  }
}

static unsigned long long fake_xgetbv(void *ctx, unsigned index) {
  fake_cpu *f = (fake_cpu*)ctx;
  ++f->xgetbv_calls;
  /* XGETBV faults unless the OS enabled XSAVE; executing it also needs AVX here */
  if (!(f->ecx1 & PFFFT_X86_ECX1_OSXSAVE) || !(f->ecx1 & PFFFT_X86_ECX1_AVX) || index != 0) f->bad = 1;
  return f->xcr0;
}

static const unsigned amd_vendor[3] = { 0x68747541u, 0x69746e65u, 0x444d4163u };   /* AuthenticAMD */
static const unsigned intel_vendor[3] = { 0x756e6547u, 0x49656e69u, 0x6c65746eu }; /* GenuineIntel */

static fake_cpu fake(const unsigned *vendor, unsigned eax1, unsigned ecx1, unsigned ebx7,
                     unsigned long long xcr0, unsigned max_leaf) {
  fake_cpu f;
  memset(&f, 0, sizeof f);
  f.max_leaf = max_leaf;
  memcpy(f.vendor, vendor, sizeof f.vendor);
  f.eax1 = eax1; f.ecx1 = ecx1; f.ebx7 = ebx7; f.xcr0 = xcr0;
  return f;
}

static int probe(fake_cpu *f, int need_fma, pffft_x86_cpu_id *id) {
  pffft_x86_cpu_ops ops;
  ops.cpuid = fake_cpuid;
  ops.xgetbv = fake_xgetbv;
  ops.ctx = f;
  return pffft_x86_probe_wide(&ops, need_fma, id);
}

static void test_probe(void) {
  const unsigned avx2 = PFFFT_X86_EBX7_AVX2;
  pffft_x86_cpu_id id;
  fake_cpu f;
  int r;

  /* Ryzen 9 5950X (Zen 3, family 19h model 21h) */
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 1 && !f.bad && f.xgetbv_calls == 1, "probe: Zen 3 supported", r, f.bad);
  check(id.vendor == PFFFT_X86_VENDOR_AMD && id.family == 0x19 && id.model == 0x21,
        "probe: Zen 3 family/model", (int)id.family, (int)id.model);
  check(pffft_x86_wide_measured(&id), "probe: Zen 3 is a measured CPU", 0, 0);

  /* Zen 4 (Raphael, model 61h) and Skylake (family 6 model 5Eh): supported, not measured */
  f = fake(amd_vendor, 0x00A60F12u, ECX_ALL, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 1 && id.model == 0x61 && !pffft_x86_wide_measured(&id), "probe: Zen 4", r, (int)id.model);
  f = fake(intel_vendor, 0x000506E3u, ECX_ALL, avx2, 0x7, 0x16);
  r = probe(&f, 1, &id);
  check(r == 1 && id.vendor == PFFFT_X86_VENDOR_INTEL && id.family == 6 && id.model == 0x5E
        && !pffft_x86_wide_measured(&id), "probe: Skylake", r, (int)id.model);

  /* missing prerequisites: no XGETBV, no out-of-range leaf */
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL & ~PFFFT_X86_ECX1_OSXSAVE, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 0 && f.xgetbv_calls == 0 && !f.bad, "probe: no OSXSAVE", r, f.xgetbv_calls);
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL & ~PFFFT_X86_ECX1_AVX, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 0 && f.xgetbv_calls == 0 && !f.bad, "probe: no AVX", r, f.xgetbv_calls);
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 0 && f.xgetbv_calls == 0 && !f.bad, "probe: no FMA", r, f.xgetbv_calls);
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, 0x7, 0x10);
  r = probe(&f, 0, &id);
  check(r == 1 && !f.bad, "probe: no FMA, not needed", r, f.bad);
  f = fake(intel_vendor, 0x000206A7u, ECX_ALL, avx2, 0x7, 0x6);   /* max leaf 6 */
  r = probe(&f, 1, &id);
  check(r == 0 && f.leaf7_calls == 0 && f.leaf1_calls == 0 && f.xgetbv_calls == 0 && !f.bad,
        "probe: max leaf 6", r, f.leaf7_calls);
  f = fake(intel_vendor, 0x000206A7u, ECX_ALL, 0, 0x7, 0xd);      /* AVX, no AVX2 */
  r = probe(&f, 1, &id);
  check(r == 0 && !f.bad, "probe: no AVX2", r, f.bad);
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x3, 0x10);    /* OS without YMM state */
  r = probe(&f, 1, &id);
  check(r == 0 && f.xgetbv_calls == 1 && !f.bad, "probe: no YMM state", r, f.xgetbv_calls);
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x1, 0x10);
  r = probe(&f, 1, NULL);
  check(r == 0 && !f.bad, "probe: XCR0 = 1, no id", r, f.bad);
}

/* ---------------- backend selection ---------------- */

static void test_selection(int have_wide) {
  static const int sizes[][2] = { { 96, PFFFT_COMPLEX }, { 32, PFFFT_REAL }, { 192, PFFFT_REAL },
                                  { 64, PFFFT_COMPLEX }, { 128, PFFFT_REAL }, { 1024, PFFFT_COMPLEX },
                                  { 4096, PFFFT_REAL } };
  size_t i;
  for (i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) {
    const int N = sizes[i][0];
    const pffft_transform_t t = (pffft_transform_t)sizes[i][1];
    const int eligible = (t == PFFFT_COMPLEX ? N % 64 : N % 128) == 0;
    PFFFT_Setup *a = pffft_new_setup(N, t), *b = pffft_new_setup(N, t);
    PFFFT_Setup *n = pffft_internal_new_setup(N, t, PFFFT_BACKEND_NATIVE);
    PFFFT_Setup *w = pffft_internal_new_setup(N, t, PFFFT_BACKEND_W8);
    check(a && b && n, "selection: setups", N, (int)t);
    if (a && b) {
      const int lanes = pffft_setup_simd_size(a);
      check(lanes == pffft_setup_simd_size(b), "selection: deterministic", N, lanes);
      check(lanes == pffft_simd_size() || (lanes == 8 && have_wide && eligible),
            "selection: lane count", N, lanes);
    }
    if (n) check(pffft_setup_simd_size(n) == pffft_simd_size(), "selection: native lanes", N, 0);
    check((w != NULL) == (have_wide && eligible), "selection: forced 8-lane", N, w != NULL);
    if (w) check(pffft_setup_simd_size(w) == 8, "selection: 8 lanes", N, pffft_setup_simd_size(w));
    if (a) pffft_destroy_setup(a);
    if (b) pffft_destroy_setup(b);
    if (n) pffft_destroy_setup(n);
    if (w) pffft_destroy_setup(w);
  }
}

/* the sizes the 8-lane backend supports, as pffft_wide_supports() has them */
static int wide_eligible(int N, pffft_transform_t t) {
  int m;
  if (N <= 0 || N % (t == PFFFT_REAL ? 128 : 64)) return 0;
  m = N / 8;
  while (m % 2 == 0) m /= 2;
  while (m % 3 == 0) m /= 3;
  while (m % 5 == 0) m /= 5;
  return m == 1;
}

static int in_table(const int *table, int count, int N) {
  int i;
  for (i = 0; i < count; ++i) if (table[i] == N) return 1;
  return 0;
}

/* The default uses 8 lanes only at table sizes, and either at all of them
   (the CPU is one of the measured ones) or at none. */
static void test_size_table(int have_wide) {
  int t, i, N;
  for (t = 0; t < 2; ++t) {
    const pffft_transform_t type = t ? PFFFT_REAL : PFFFT_COMPLEX;
    const int *table = t ? pffft_wide_real_sizes : pffft_wide_complex_sizes;
    const int count = t ? PFFFT_WIDE_REAL_COUNT : PFFFT_WIDE_COMPLEX_COUNT;
    int wide_count = 0, eligible = 0, checked = 0;
    for (i = 0; i < count; ++i) {
      check(wide_eligible(table[i], type), "size table: entry the backend cannot handle", table[i], t);
      if (i) check(table[i] > table[i-1], "size table: not increasing", table[i], t);
    }
    for (N = 64; N <= (1 << 20); N += 64) {
      PFFFT_Setup *s;
      int lanes;
      if (!wide_eligible(N, type)) continue;
      /* large setups are slow under emulation: above 2^16 check the table
         entries and every 8th other size */
      if (N > (1 << 16) && !in_table(table, count, N) && eligible++ % 8) continue;
      if (N <= (1 << 16)) ++eligible;
      ++checked;
      s = pffft_new_setup(N, type);
      if (!s) { check(0, "size table: setup", N, t); continue; }
      lanes = pffft_setup_simd_size(s);
      pffft_destroy_setup(s);
      if (lanes == 8) ++wide_count;
      if (in_table(table, count, N)) check(lanes == 8 || lanes == pffft_simd_size(), "size table: lanes", N, lanes);
      else check(lanes == pffft_simd_size(), "size table: 8 lanes at a size not in the table", N, lanes);
    }
    check(wide_count == 0 || (have_wide && wide_count == count), "size table: 8 lanes not at exactly all or none of the table sizes", wide_count, count);
    printf("%s: default uses 8 lanes at %d of %d checked sizes (table: %d)\n",
           t ? "REAL" : "COMPLEX", wide_count, checked, count);
  }
}

/* ---------------- alignment ---------------- */

#define GUARD 64          /* floats of guard region on each side */
#define GUARD_VALUE 12345.0f

typedef struct {
  float *base;   /* 64-byte aligned allocation */
  float *p;      /* the buffer: base + GUARD + offset */
  int n, offset;
} guarded;

static guarded guarded_new(int n, int offset) {
  guarded g;
  int i;
  g.n = n; g.offset = offset;
  g.base = (float*)pffft_aligned_malloc(sizeof(float) * (size_t)(n + 2*GUARD + 8));
  if (!g.base) { printf("out of memory\n"); exit(2); }
  for (i = 0; i < n + 2*GUARD + 8; ++i) g.base[i] = GUARD_VALUE;
  g.p = g.base + GUARD + offset;
  return g;
}

static int guarded_intact(const guarded *g) {
  int i;
  for (i = 0; i < GUARD + g->offset; ++i) if (g->base[i] != GUARD_VALUE) return 0;
  for (i = GUARD + g->offset + g->n; i < g->n + 2*GUARD + 8; ++i) if (g->base[i] != GUARD_VALUE) return 0;
  return 1;
}

static void fill(float *x, int n, unsigned seed) {
  int i;
  for (i = 0; i < n; ++i) {
    seed = seed * 1664525u + 1013904223u;
    x[i] = (float)((int)(seed >> 9) - (1 << 22)) / (float)(1 << 22);
  }
}

/* the eight transform modes, then the spectral operations */
enum { OP_TRANSFORM = 0, NUM_TRANSFORM_OPS = 8,
       OP_REORDER_FWD = NUM_TRANSFORM_OPS, OP_REORDER_BWD, OP_ZCONV, OP_ZCONV_SCALE,
       OP_ZCONV_ACC, OP_ZP, OP_ZCONV_ZP, NUM_OPS };
#define OP_IS_BACKWARD(op) ((op) & 1)
#define OP_IS_ORDERED(op) ((op) & 2)
#define OP_IS_INPLACE(op) ((op) & 4)

static const char *op_names[NUM_OPS] = {
  "forward", "backward", "ordered forward", "ordered backward",
  "in-place forward", "in-place backward", "in-place ordered forward", "in-place ordered backward",
  "zreorder forward", "zreorder backward", "zconvolve", "zconvolve_scale",
  "zconvolve_accumulate", "zconvert_zp", "zconvolve_zp" };

/* operands: a (input), b (second spectral operand), o (output) */
enum { ALIAS_NONE, ALIAS_OA, ALIAS_OB, ALIAS_AB, ALIAS_ALL, NUM_ALIAS };
static const char *alias_names[NUM_ALIAS] = { "", " out=a", " out=b", " a=b", " out=a=b" };

static int op_operands(int op) {   /* 1: a only, 2: a and b */
  return op == OP_ZCONV || op == OP_ZCONV_SCALE || op == OP_ZCONV_ACC || op == OP_ZCONV_ZP ? 2 : 1;
}

/* whether op supports the alias pattern; transforms take their in-place
   mode from the op, zreorder must not alias */
static int alias_allowed(int op, int alias) {
  if (op < NUM_TRANSFORM_OPS) return alias == (OP_IS_INPLACE(op) ? ALIAS_OA : ALIAS_NONE);
  if (op == OP_REORDER_FWD || op == OP_REORDER_BWD) return alias == ALIAS_NONE;
  if (op_operands(op) == 1) return alias == ALIAS_NONE || alias == ALIAS_OA;
  return 1;
}

/* run op; returns 0 if the op does not apply to this transform type */
static int run_op(PFFFT_Setup *s, int op, int real, const float *a, const float *b, float *o, float *work) {
  if (op < NUM_TRANSFORM_OPS) {
    const pffft_direction_t dir = OP_IS_BACKWARD(op) ? PFFFT_BACKWARD : PFFFT_FORWARD;
    if (OP_IS_ORDERED(op)) pffft_transform_ordered(s, a, o, work, dir);
    else pffft_transform(s, a, o, work, dir);
    return 1;
  }
  switch (op) {
    case OP_REORDER_FWD: pffft_zreorder(s, a, o, PFFFT_FORWARD); return 1;
    case OP_REORDER_BWD: pffft_zreorder(s, a, o, PFFFT_BACKWARD); return 1;
    case OP_ZCONV: pffft_zconvolve(s, a, b, o); return 1;
    case OP_ZCONV_SCALE: pffft_zconvolve_scale(s, a, b, o, 0.25f); return 1;
    case OP_ZCONV_ACC: pffft_zconvolve_accumulate(s, a, b, o, 0.25f); return 1;
    case OP_ZP: return real ? (pffft_zconvert_zp(s, a, o, 2.0f) == 0) : 0;
    case OP_ZCONV_ZP: return real ? (pffft_zconvolve_zp(s, a, b, o) == 0) : 0;
    default: return 0;
  }
}

static int alignment_cases;

/* Every op with every alias pattern it allows, each distinct buffer at a
   16- or 64-byte aligned address (offset 4 or 0 floats) and work = NULL
   or not, compared bitwise with the same op on separate 64-byte aligned
   buffers holding the same values. */
static void test_alignment_size(int N, pffft_transform_t t) {
  const int real = t == PFFFT_REAL, nf = real ? N : 2*N;
  const size_t nbytes = sizeof(float) * (size_t)nf;
  PFFFT_Setup *s = pffft_internal_new_setup(N, t, PFFFT_BACKEND_W8);
  float *x = (float*)pffft_aligned_malloc(nbytes), *y = (float*)pffft_aligned_malloc(nbytes);
  float *z = (float*)pffft_aligned_malloc(nbytes);
  float *ra = (float*)pffft_aligned_malloc(nbytes), *rb = (float*)pffft_aligned_malloc(nbytes);
  float *ref = (float*)pffft_aligned_malloc(nbytes), *work = (float*)pffft_aligned_malloc(nbytes);
  int op, alias, offs;
  if (!s || !x || !y || !z || !ra || !rb || !ref || !work) {
    check(0, "alignment: setup/allocation", N, (int)t);
    goto done;
  }
  fill(x, nf, 1u + (unsigned)N);
  fill(y, nf, 7u + (unsigned)N);
  fill(z, nf, 13u + (unsigned)N);

  for (op = 0; op < NUM_OPS; ++op) {
    const int two = op_operands(op) == 2, transform = op < NUM_TRANSFORM_OPS;
    for (alias = 0; alias < NUM_ALIAS; ++alias) {
      /* the values the operands start with: aliased ones share them */
      const int o_is_a = alias == ALIAS_OA || alias == ALIAS_ALL, o_is_b = alias == ALIAS_OB;
      const int b_is_a = alias == ALIAS_AB || alias == ALIAS_ALL;
      const float *va = x, *vb = b_is_a ? x : y, *vo = o_is_a ? x : o_is_b ? y : z;
      if (!alias_allowed(op, alias) || (!two && (alias == ALIAS_OB || alias == ALIAS_AB || alias == ALIAS_ALL)))
        continue;
      memcpy(ra, va, nbytes); memcpy(rb, vb, nbytes); memcpy(ref, vo, nbytes);
      if (!run_op(s, op, real, ra, rb, ref, work)) break;   /* does not apply to this type */

      /* offset bits: a, b, o, work; only for distinct buffers that exist */
      for (offs = 0; offs < 16; ++offs) {
        const int oa = (offs & 1) ? 4 : 0, ob = (offs & 2) ? 4 : 0, oo = (offs & 4) ? 4 : 0, ow = (offs & 8) ? 4 : 0;
        guarded ga, gb, go, gw;
        int use_null_work;
        if (ob && (!two || b_is_a)) continue;
        if (oo && (o_is_a || o_is_b)) continue;
        if (ow && !transform) continue;
        ga = guarded_new(nf, oa); gb = guarded_new(nf, ob); go = guarded_new(nf, oo); gw = guarded_new(nf, ow);
        for (use_null_work = 0; use_null_work < 1 + transform; ++use_null_work) {
          float *pa = ga.p, *pb = b_is_a ? ga.p : gb.p, *po = o_is_a ? ga.p : o_is_b ? gb.p : go.p;
          int ok;
          memcpy(ga.p, va, nbytes); memcpy(gb.p, vb, nbytes); memcpy(go.p, vo, nbytes);
          run_op(s, op, real, pa, pb, po, use_null_work ? NULL : gw.p);
          ++alignment_cases;
          ok = memcmp(po, ref, nbytes) == 0;
          /* operands that are not the output are left untouched */
          if (pa != po) ok = ok && memcmp(pa, va, nbytes) == 0;
          if (two && pb != po && pb != pa) ok = ok && memcmp(pb, vb, nbytes) == 0;
          if (!ok || !guarded_intact(&ga) || !guarded_intact(&gb) || !guarded_intact(&go) || !guarded_intact(&gw)) {
            ++failures;
            printf("FAIL: alignment: %s %s%s N=%d offsets a/b/out/work %d/%d/%d/%d%s%s\n",
                   real ? "REAL" : "COMPLEX", op_names[op], alias_names[alias], N, 4*oa, 4*ob, 4*oo, 4*ow,
                   use_null_work ? " work=NULL" : "", ok ? " (guard region overwritten)" : "");
          }
        }
        pffft_aligned_free(ga.base); pffft_aligned_free(gb.base);
        pffft_aligned_free(go.base); pffft_aligned_free(gw.base);
      }
    }
  }
done:
  if (s) pffft_destroy_setup(s);
  pffft_aligned_free(x); pffft_aligned_free(y); pffft_aligned_free(z);
  pffft_aligned_free(ra); pffft_aligned_free(rb); pffft_aligned_free(ref); pffft_aligned_free(work);
}

static void test_alignment(void) {
  static const int complex_sizes[] = { 64, 192, 320, 1024, 1536, 4096 };
  static const int real_sizes[] = { 128, 384, 640, 2048, 3072, 8192 };
  size_t i;
  for (i = 0; i < sizeof complex_sizes / sizeof complex_sizes[0]; ++i)
    test_alignment_size(complex_sizes[i], PFFFT_COMPLEX);
  for (i = 0; i < sizeof real_sizes / sizeof real_sizes[0]; ++i)
    test_alignment_size(real_sizes[i], PFFFT_REAL);
  printf("alignment: %d cases\n", alignment_cases);
}

/* ---------------- concurrency ---------------- */

#ifdef PFFFT_TEST_THREADS
#define NUM_THREADS 8
#define THREAD_ROUNDS 200

static float *thread_ref[2];

static void *thread_main(void *arg) {
  const int N = 1024;
  float *x = (float*)pffft_aligned_malloc(sizeof(float) * 2 * N);
  float *out = (float*)pffft_aligned_malloc(sizeof(float) * 2 * N);
  int round, *bad = (int*)arg;
  if (!x || !out) { *bad = 1; return NULL; }
  fill(x, 2 * N, 99u);
  for (round = 0; round < THREAD_ROUNDS; ++round) {
    const int cplx = round & 1;
    PFFFT_Setup *s = pffft_new_setup(N, cplx ? PFFFT_COMPLEX : PFFFT_REAL);
    if (!s) { *bad = 1; break; }
    pffft_transform_ordered(s, x, out, NULL, PFFFT_FORWARD);
    if (memcmp(out, thread_ref[cplx], sizeof(float) * (size_t)(cplx ? 2 * N : N)) != 0) *bad = 1;
    pffft_destroy_setup(s);
  }
  pffft_aligned_free(x); pffft_aligned_free(out);
  return NULL;
}

static void test_threads(void) {
  const int N = 1024;
  pthread_t threads[NUM_THREADS];
  int bad[NUM_THREADS], i, cplx;
  float *x = (float*)pffft_aligned_malloc(sizeof(float) * 2 * N);
  if (!x) { check(0, "threads: allocation", 0, 0); return; }
  fill(x, 2 * N, 99u);
  for (cplx = 0; cplx < 2; ++cplx) {
    PFFFT_Setup *s = pffft_new_setup(N, cplx ? PFFFT_COMPLEX : PFFFT_REAL);
    thread_ref[cplx] = (float*)pffft_aligned_malloc(sizeof(float) * 2 * N);
    if (!s || !thread_ref[cplx]) { check(0, "threads: setup", cplx, 0); return; }
    pffft_transform_ordered(s, x, thread_ref[cplx], NULL, PFFFT_FORWARD);
    pffft_destroy_setup(s);
  }
  for (i = 0; i < NUM_THREADS; ++i) {
    bad[i] = 0;
    if (pthread_create(&threads[i], NULL, thread_main, &bad[i]) != 0) {
      check(0, "threads: pthread_create", i, 0);
      bad[i] = -1;
    }
  }
  for (i = 0; i < NUM_THREADS; ++i) {
    if (bad[i] != -1) pthread_join(threads[i], NULL);
    check(bad[i] == 0, "threads: concurrent setup/transform results", i, bad[i]);
  }
  pffft_aligned_free(thread_ref[0]); pffft_aligned_free(thread_ref[1]);
  pffft_aligned_free(x);
}
#endif

int main(void) {
  PFFFT_Setup *probe = pffft_internal_new_setup(128, PFFFT_REAL, PFFFT_BACKEND_W8);
  const int have_wide = probe != NULL;
  if (probe) pffft_destroy_setup(probe);
  printf("8-lane backend %s\n", have_wide ? "available: testing it" : "not available in this build or on this CPU");

  test_decode();
  test_probe();
  test_selection(have_wide);
  test_size_table(have_wide);
  if (have_wide) test_alignment();
#ifdef PFFFT_TEST_THREADS
  test_threads();
#endif

  if (failures) {
    printf("%d FAILURES\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
