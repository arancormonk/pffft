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
#include "pffft_wide_select.h"

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

/* ---------------- CPU detection ---------------- */

#define ECX_ALL (PFFFT_X86_ECX1_SSE3 | PFFFT_X86_ECX1_SSSE3 | PFFFT_X86_ECX1_SSE41 | \
                 PFFFT_X86_ECX1_SSE42 | PFFFT_X86_ECX1_POPCNT | PFFFT_X86_ECX1_XSAVE | \
                 PFFFT_X86_ECX1_OSXSAVE | PFFFT_X86_ECX1_AVX | PFFFT_X86_ECX1_FMA | \
                 PFFFT_X86_ECX1_MOVBE | PFFFT_X86_ECX1_F16C)

#define ECX81_ALL PFFFT_X86_ECX81_LZCNT

static void test_decode(void) {
  const unsigned avx2 = PFFFT_X86_EBX7_AVX2 | PFFFT_X86_EBX7_BMI1 | PFFFT_X86_EBX7_BMI2;
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, ECX81_ALL, 0x7, 1) == 1, "decode: all features", 0, 0);
  check(pffft_x86_decode_wide(0xd, ECX_ALL, avx2, ECX81_ALL, 0xe7, 1) == 1, "decode: AVX-512 state too", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, ECX81_ALL, 0x7, 1) == 0, "decode: no FMA", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_FMA, avx2, ECX81_ALL, 0x7, 0) == 1, "decode: no FMA, not needed", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_AVX, avx2, ECX81_ALL, 0x7, 1) == 0, "decode: no AVX", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL & ~PFFFT_X86_ECX1_OSXSAVE, avx2, ECX81_ALL, 0x7, 1) == 0, "decode: no OSXSAVE", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, 0, ECX81_ALL, 0x7, 1) == 0, "decode: no AVX2", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, ECX81_ALL, 0x5, 1) == 0, "decode: no XMM state", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, ECX81_ALL, 0x3, 1) == 0, "decode: no YMM state", 0, 0);
  check(pffft_x86_decode_wide(7, ECX_ALL, avx2, ECX81_ALL, 0x1, 1) == 0, "decode: XCR0 = 1", 0, 0);
  check(pffft_x86_decode_wide(6, ECX_ALL, avx2, ECX81_ALL, 0x7, 1) == 0, "decode: max leaf 6", 0, 0);
  {
    /* every extension the 8-lane code may use is required (the ISA audit
       allows exactly these in pffft_w8_* functions) */
    int k;
    for (k = 0; k < PFFFT_X86_WIDE_ISA_COUNT; ++k) {
      const pffft_x86_isa_entry *e = &pffft_x86_wide_isa[k];
      unsigned ecx = ECX_ALL & ~(e->reg == 1 ? e->bit : 0u), ebx = avx2 & ~(e->reg == 7 ? e->bit : 0u);
      int fma = e->reg == 1 && e->bit == PFFFT_X86_ECX1_FMA;
      check(e->reg == 1 ? (ECX_ALL & e->bit) != 0 : e->reg == 7 ? (avx2 & e->bit) != 0 : (ECX81_ALL & e->bit) != 0,
            "ISA table entry in the test's feature set", k, 0);
      const unsigned e81 = ECX81_ALL & ~(e->reg == 0x81 ? e->bit : 0u);
      check(pffft_x86_decode_wide(7, ecx, ebx, e81, 0x7, 1) == 0, "decode: ISA table entry missing", k, 0);
      check(pffft_x86_decode_wide(7, ecx, ebx, e81, 0x7, 0) == fma, "decode: ISA table entry missing, no FMA needed", k, 0);
    }
  }
}

typedef struct {
  unsigned max_leaf, vendor[3], eax1, ecx1, ebx7, edx7;
  unsigned long long xcr0;
  /* deterministic cache leaves (4 / 8000_001Dh): eax, ebx, ecx, edx per subleaf */
  unsigned leaf4[4][4], leaf1d[4][4];
  int n_leaf4, n_leaf1d;
  unsigned max_ext, ecx81, ecx86;
  int xgetbv_calls, leaf1_calls, leaf7_calls, bad;
} fake_cpu;

static void fake_cpuid(void *ctx, unsigned leaf, unsigned subleaf, unsigned *r) {
  fake_cpu *f = (fake_cpu*)ctx;
  r[0] = r[1] = r[2] = r[3] = 0;
  if (leaf >= 0x80000000u) {
    /* 8000_0000h itself is always valid */
    if (leaf != 0x80000000u && leaf > f->max_ext) f->bad = 1;
    if (leaf == 0x80000000u) r[0] = f->max_ext;
    else if (leaf == 0x80000001u) r[2] = f->ecx81;
    else if (leaf == 0x80000006u) r[2] = f->ecx86;
    else if (leaf == 0x8000001Du && (int)subleaf < f->n_leaf1d) memcpy(r, f->leaf1d[subleaf], 4 * sizeof *r);
    return;
  }
  if (leaf > f->max_leaf) f->bad = 1;   /* unsupported leaf queried */
  if (leaf == 0) {
    r[0] = f->max_leaf; r[1] = f->vendor[0]; r[3] = f->vendor[1]; r[2] = f->vendor[2];
  } else if (leaf == 1) {
    ++f->leaf1_calls;
    r[0] = f->eax1; r[2] = f->ecx1;
  } else if (leaf == 4) {
    if ((int)subleaf < f->n_leaf4) memcpy(r, f->leaf4[subleaf], 4 * sizeof *r);
  } else if (leaf == 7) {
    ++f->leaf7_calls;
    if (subleaf == 0) { r[1] = f->ebx7; r[3] = f->edx7; }
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
  f.max_ext = 0x80000008u; f.ecx81 = PFFFT_X86_ECX81_LZCNT;
  return f;
}

static pffft_x86_cpu_ops fake_ops(fake_cpu *f) {
  pffft_x86_cpu_ops ops;
  ops.cpuid = fake_cpuid;
  ops.xgetbv = fake_xgetbv;
  ops.ctx = f;
  return ops;
}

static int probe(fake_cpu *f, int need_fma, pffft_x86_cpu_id *id) {
  pffft_x86_cpu_ops ops = fake_ops(f);
  return pffft_x86_probe_wide(&ops, need_fma, id);
}

static void test_probe(void) {
  const unsigned avx2 = PFFFT_X86_EBX7_AVX2 | PFFFT_X86_EBX7_BMI1 | PFFFT_X86_EBX7_BMI2;
  pffft_x86_cpu_id id;
  fake_cpu f;
  int r;

  /* Ryzen 9 5950X (Zen 3, family 19h model 21h) */
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 1 && !f.bad && f.xgetbv_calls == 1, "probe: Zen 3 supported", r, f.bad);
  check(id.vendor == PFFFT_X86_VENDOR_AMD && id.family == 0x19 && id.model == 0x21,
        "probe: Zen 3 family/model", (int)id.family, (int)id.model);
  check(pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_ZEN3_DESKTOP, "probe: Zen 3 desktop is measured", 0, 0);
  /* EPYC 7763 (Milan, family 19h model 01h): measured, its own table */
  f = fake(amd_vendor, 0x00A00F11u, ECX_ALL, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 1 && id.model == 0x01 && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_ZEN3_SERVER,
        "probe: Zen 3 server", r, (int)id.model);

  /* Zen 4 (Raphael, model 61h): measured; Skylake (family 6 model 5Eh): inferred */
  f = fake(amd_vendor, 0x00A60F12u, ECX_ALL, avx2, 0x7, 0x10);
  r = probe(&f, 1, &id);
  check(r == 1 && id.model == 0x61 && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_ZEN4,
        "probe: Zen 4", r, (int)id.model);
  f = fake(intel_vendor, 0x000506E3u, ECX_ALL, avx2, 0x7, 0x16);
  r = probe(&f, 1, &id);
  check(r == 1 && id.vendor == PFFFT_X86_VENDOR_INTEL && id.family == 6 && id.model == 0x5E
        && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_INFERRED, "probe: Skylake", r, (int)id.model);

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

  /* 06_9A stepping 4 with Hybrid clear: Arizona Beach (Gracemont E-cores only) */
  f = fake(intel_vendor, 0x000906A4u, ECX_ALL, avx2, 0x7, 0x20);
  r = probe(&f, 1, &id);
  check(r == 1 && id.model == 0x9A && !id.hybrid && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_OFF,
        "probe: 906A4 Gracemont-only is off", r, (int)id.model);
  /* Alder Lake-S 06_97 with and without the Hybrid bit */
  f = fake(intel_vendor, 0x00090672u, ECX_ALL, avx2, 0x7, 0x20);
  r = probe(&f, 1, &id);
  check(r == 1 && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_INFERRED, "probe: 06_97 P-cores only", r, id.hybrid);
  f.edx7 = PFFFT_X86_EDX7_HYBRID;
  r = probe(&f, 1, &id);
  check(r == 1 && id.hybrid && pffft_x86_wide_class(&id) == PFFFT_X86_WIDE_OFF, "probe: 06_97 hybrid", r, id.hybrid);
}

/* ---------------- CPU classes ---------------- */

static int cls(int vendor, unsigned family, unsigned model, int hybrid) {
  pffft_x86_cpu_id id;
  id.vendor = vendor; id.family = family; id.model = model; id.hybrid = hybrid; id.max_leaf = 0x20;
  return pffft_x86_wide_class(&id);
}

static void test_classes(void) {
  enum { O = PFFFT_X86_WIDE_OFF, I = PFFFT_X86_WIDE_INFERRED, M = PFFFT_X86_WIDE_ZEN3,
         Z4 = PFFFT_X86_WIDE_ZEN4, ZS = PFFFT_X86_WIDE_ZEN3_SERVER, ZD = PFFFT_X86_WIDE_ZEN3_DESKTOP };
  static const struct { unsigned family, model; int expect; } amd[] = {
    /* Zen 3: server and desktop measured, the APUs with both models' sizes; Badami inferred */
    { 0x19, 0x00, ZS }, { 0x19, 0x01, ZS }, { 0x19, 0x08, ZS }, { 0x19, 0x0F, ZS },
    { 0x19, 0x20, ZD }, { 0x19, 0x21, ZD }, { 0x19, 0x2F, ZD },
    { 0x19, 0x40, M }, { 0x19, 0x4F, M }, { 0x19, 0x50, M }, { 0x19, 0x5F, M },
    { 0x19, 0x30, I }, { 0x19, 0x3F, I },
    /* Zen 4 (measured) and the holes around it */
    { 0x19, 0x10, Z4 }, { 0x19, 0x11, Z4 }, { 0x19, 0x1F, Z4 }, { 0x19, 0x60, Z4 }, { 0x19, 0x61, Z4 },
    { 0x19, 0x7F, Z4 }, { 0x19, 0xA0, Z4 }, { 0x19, 0xAF, Z4 }, { 0x19, 0x80, O }, { 0x19, 0x9F, O },
    { 0x19, 0xB0, O }, { 0x19, 0xFF, O },
    /* family 17h: Zen and Zen+ off, Zen 2 ranges inferred, holes off */
    { 0x17, 0x00, O }, { 0x17, 0x01, O }, { 0x17, 0x08, O }, { 0x17, 0x11, O }, { 0x17, 0x18, O },
    { 0x17, 0x20, O }, { 0x17, 0x2F, O }, { 0x17, 0x30, I }, { 0x17, 0x31, I }, { 0x17, 0x3F, I },
    { 0x17, 0x40, O }, { 0x17, 0x46, O }, { 0x17, 0x47, I }, { 0x17, 0x48, O }, { 0x17, 0x4F, O },
    { 0x17, 0x50, O }, { 0x17, 0x5F, O }, { 0x17, 0x60, I }, { 0x17, 0x68, I }, { 0x17, 0x71, I },
    { 0x17, 0x7F, I }, { 0x17, 0x80, O }, { 0x17, 0x83, O }, { 0x17, 0x84, I }, { 0x17, 0x87, I },
    { 0x17, 0x88, O }, { 0x17, 0x8F, O }, { 0x17, 0x90, I }, { 0x17, 0xA0, I }, { 0x17, 0xAF, I },
    { 0x17, 0xB0, O },
    /* Zen 5 and its holes */
    { 0x1A, 0x00, I }, { 0x1A, 0x44, I }, { 0x1A, 0x4F, I }, { 0x1A, 0x50, O }, { 0x1A, 0x5F, O },
    { 0x1A, 0x60, I }, { 0x1A, 0x77, I }, { 0x1A, 0x78, O }, { 0x1A, 0xCF, O }, { 0x1A, 0xD0, I },
    { 0x1A, 0xD7, I }, { 0x1A, 0xD8, O },
    /* Excavator, Jaguar, family 18h, future families */
    { 0x15, 0x60, O }, { 0x16, 0x30, O }, { 0x18, 0x00, O }, { 0x1B, 0x00, O }, { 0x20, 0x00, O }
  };
  static const unsigned char intel_inferred[] = { 0x3C, 0x45, 0x46, 0x3D, 0x47, 0x4E, 0x5E, 0x8E, 0x9E,
                                                  0xA5, 0xA6, 0x66, 0x7D, 0x7E, 0x8C, 0x8D, 0xA7 };
  /* server, Atom-only, always-hybrid, Xeon Phi, neighbours of the listed models */
  static const unsigned char intel_off[] = { 0x3F, 0x4F, 0x56, 0x55, 0x6A, 0x6C, 0x8F, 0xCF, 0xAD, 0xAE,
                                             0xBE, 0xAF, 0xB6, 0xDD, 0x57, 0x85, 0x9A, 0xBA, 0xAA, 0xAC,
                                             0xB5, 0xC5, 0xC6, 0xBD, 0xCC, 0xD5, 0x3B, 0x3E, 0x44, 0x48,
                                             0x4D, 0x5D, 0x5F, 0x65, 0x67, 0x7C, 0x7F, 0x8B, 0x9D, 0x9F,
                                             0xA4, 0xA8, 0x96, 0x98, 0xB8, 0xC0 };
  static const unsigned char intel_pcore_only[] = { 0x97, 0xB7, 0xBF };
  size_t k;
  for (k = 0; k < sizeof amd / sizeof amd[0]; ++k)
    check(cls(PFFFT_X86_VENDOR_AMD, amd[k].family, amd[k].model, 0) == amd[k].expect,
          "class: AMD family/model", (int)amd[k].family, (int)amd[k].model);
  for (k = 0; k < sizeof intel_inferred; ++k) {
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_inferred[k], 0) == I, "class: Intel inferred", 6, intel_inferred[k]);
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_inferred[k], 1) == O, "class: Intel hybrid bit", 6, intel_inferred[k]);
  }
  for (k = 0; k < sizeof intel_off; ++k) {
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_off[k], 0) == O, "class: Intel off", 6, intel_off[k]);
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_off[k], 1) == O, "class: Intel off, hybrid", 6, intel_off[k]);
  }
  for (k = 0; k < sizeof intel_pcore_only; ++k) {
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_pcore_only[k], 0) == I, "class: P-core-only", 6, intel_pcore_only[k]);
    check(cls(PFFFT_X86_VENDOR_INTEL, 6, intel_pcore_only[k], 1) == O, "class: hybrid", 6, intel_pcore_only[k]);
  }
  /* Intel families beyond 6 (Diamond Rapids 13h, Nova Lake 12h), Pentium 4, other vendors */
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x13, 0x01, 0) == O, "class: Intel family 13h", 0x13, 1);
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x12, 0x01, 0) == O, "class: Intel family 12h", 0x12, 1);
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x0F, 0x06, 0) == O, "class: Intel family 0Fh", 0x0F, 6);
  /* listed model numbers count only in family 6 (AMD: only in their family) */
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x13, 0x5E, 0) == O, "class: Intel family 13h, model 5Eh", 0x13, 0x5E);
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x12, 0x97, 0) == O, "class: Intel family 12h, model 97h", 0x12, 0x97);
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x05, 0x3C, 0) == O, "class: Intel family 5, model 3Ch", 5, 0x3C);
  check(cls(PFFFT_X86_VENDOR_AMD, 0x06, 0x3C, 0) == O, "class: AMD family 6, model 3Ch", 6, 0x3C);
  check(cls(PFFFT_X86_VENDOR_INTEL, 0x19, 0x21, 0) == O, "class: Intel, Zen 3 numbers", 0x19, 0x21);
  check(cls(PFFFT_X86_VENDOR_OTHER, 0x18, 0x00, 0) == O, "class: Hygon", 0x18, 0);
  check(cls(PFFFT_X86_VENDOR_OTHER, 0x19, 0x21, 0) == O, "class: other vendor, Zen 3 numbers", 0x19, 0x21);
  check(cls(PFFFT_X86_VENDOR_OTHER, 6, 0x5E, 0) == O, "class: other vendor, Skylake numbers", 6, 0x5E);
}

/* ---------------- L2 size ---------------- */

/* a deterministic cache leaf entry: type (1 data, 2 instruction, 3 unified),
   level, ways, partitions, line size, sets, stored as the leaves do */
static void cache_entry(unsigned *r, unsigned type, unsigned level, unsigned ways, unsigned partitions,
                        unsigned line, unsigned sets) {
  r[0] = type | (level << 5);
  r[1] = (ways - 1) << 22 | (partitions - 1) << 12 | (line - 1);
  r[2] = sets - 1;
  r[3] = 0;
}

static unsigned long long l2_of(fake_cpu *f, int *bad) {
  pffft_x86_cpu_ops ops = fake_ops(f);
  pffft_x86_cpu_id id;
  unsigned long long l2;
  pffft_x86_probe_wide(&ops, 1, &id);
  l2 = pffft_x86_l2_bytes(&ops, &id);
  *bad = f->bad;
  return l2;
}

static void test_l2(void) {
  const unsigned avx2 = PFFFT_X86_EBX7_AVX2 | PFFFT_X86_EBX7_BMI1 | PFFFT_X86_EBX7_BMI2;
  fake_cpu f;
  unsigned long long l2;
  int bad;

  /* Intel: leaf 4 (i7-1065G7: 48 KiB L1d, 32 KiB L1i, 512 KiB L2, 8 MiB L3); the
     legacy 8000_0006h value (256 KiB) is not used */
  f = fake(intel_vendor, 0x000706E5u, ECX_ALL, avx2, 0x7, 0x1B);
  cache_entry(f.leaf4[0], 1, 1, 12, 1, 64, 64);
  cache_entry(f.leaf4[1], 2, 1, 8, 1, 64, 64);
  cache_entry(f.leaf4[2], 3, 2, 8, 1, 64, 1024);
  cache_entry(f.leaf4[3], 3, 3, 16, 1, 64, 8192);
  f.n_leaf4 = 4; f.max_ext = 0x80000008u; f.ecx86 = 256u << 16;
  l2 = l2_of(&f, &bad);
  check(l2 == 512u * 1024 && !bad, "L2: Intel leaf 4", (int)(l2 >> 10), bad);
  /* raw partition count 0 is one partition; two partitions double it */
  cache_entry(f.leaf4[2], 3, 2, 8, 2, 64, 1024);
  l2 = l2_of(&f, &bad);
  check(l2 == 1024u * 1024 && !bad, "L2: Intel two partitions", (int)(l2 >> 10), bad);
  /* no level-2 entry, or the list ends at once */
  cache_entry(f.leaf4[2], 3, 3, 16, 1, 64, 8192);
  f.n_leaf4 = 3;
  l2 = l2_of(&f, &bad);
  check(l2 == 0 && !bad, "L2: Intel without L2 entry", (int)(l2 >> 10), bad);
  f.n_leaf4 = 0;
  l2 = l2_of(&f, &bad);
  check(l2 == 0 && !bad, "L2: Intel empty leaf 4", (int)(l2 >> 10), bad);

  /* AMD: leaf 8000_001Dh with TOPOEXT (Zen 3: 512 KiB L2) */
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x7, 0x10);
  cache_entry(f.leaf1d[0], 1, 1, 8, 1, 64, 64);
  cache_entry(f.leaf1d[1], 2, 1, 8, 1, 64, 64);
  cache_entry(f.leaf1d[2], 3, 2, 8, 1, 64, 1024);
  f.n_leaf1d = 3; f.max_ext = 0x80000023u; f.ecx81 = PFFFT_X86_ECX81_TOPOEXT; f.ecx86 = 1024u << 16;
  l2 = l2_of(&f, &bad);
  check(l2 == 512u * 1024 && !bad, "L2: AMD leaf 8000_001Dh", (int)(l2 >> 10), bad);
  /* invalid 8000_001Dh: the legacy leaf */
  f.n_leaf1d = 0;
  l2 = l2_of(&f, &bad);
  check(l2 == 1024u * 1024 && !bad, "L2: AMD fallback to 8000_0006h", (int)(l2 >> 10), bad);
  /* no TOPOEXT: the legacy leaf, and 8000_001Dh is not queried */
  f.n_leaf1d = 3; f.ecx81 = 0;
  l2 = l2_of(&f, &bad);
  check(l2 == 1024u * 1024 && !bad, "L2: AMD without TOPOEXT", (int)(l2 >> 10), bad);
  /* TOPOEXT but 8000_001Dh beyond the maximum extended leaf */
  f.ecx81 = PFFFT_X86_ECX81_TOPOEXT; f.max_ext = 0x80000008u;
  l2 = l2_of(&f, &bad);
  check(l2 == 1024u * 1024 && !bad, "L2: AMD 8000_001Dh not available", (int)(l2 >> 10), bad);
  /* maximum extended leaf below 8000_0006h, or not an extended leaf number */
  f.max_ext = 0x80000005u;
  l2 = l2_of(&f, &bad);
  check(l2 == 0 && !bad, "L2: AMD without 8000_0006h", (int)(l2 >> 10), bad);
  f.max_ext = 0x0000000Fu;
  l2 = l2_of(&f, &bad);
  check(l2 == 0 && !bad, "L2: AMD invalid 8000_0000h", (int)(l2 >> 10), bad);
  /* other vendors: unknown */
  f = fake(amd_vendor, 0x00A20F10u, ECX_ALL, avx2, 0x7, 0x10);
  f.vendor[0] = 0x6f677948u;   /* "Hygo..." */
  f.max_ext = 0x80000008u; f.ecx86 = 512u << 16;
  l2 = l2_of(&f, &bad);
  check(l2 == 0 && !bad, "L2: other vendor", (int)(l2 >> 10), bad);
}

/* ---------------- default selection rule ---------------- */

/* the measured table of a CPU class, NULL for the others */
static const int *measured_table(int cls, int real, int *count) {
  if (cls == PFFFT_X86_WIDE_ZEN3) {
    *count = real ? PFFFT_WIDE_ZEN3_REAL_COUNT : PFFFT_WIDE_ZEN3_COMPLEX_COUNT;
    return real ? pffft_wide_zen3_real_sizes : pffft_wide_zen3_complex_sizes;
  }
  if (cls == PFFFT_X86_WIDE_ZEN3_DESKTOP) {
    *count = real ? PFFFT_WIDE_ZEN3_DESKTOP_REAL_COUNT : PFFFT_WIDE_ZEN3_DESKTOP_COMPLEX_COUNT;
    return real ? pffft_wide_zen3_desktop_real_sizes : pffft_wide_zen3_desktop_complex_sizes;
  }
  if (cls == PFFFT_X86_WIDE_ZEN3_SERVER) {
    *count = real ? PFFFT_WIDE_ZEN3_SERVER_REAL_COUNT : PFFFT_WIDE_ZEN3_SERVER_COMPLEX_COUNT;
    return real ? pffft_wide_zen3_server_real_sizes : pffft_wide_zen3_server_complex_sizes;
  }
  if (cls == PFFFT_X86_WIDE_ZEN4) {
    *count = real ? PFFFT_WIDE_ZEN4_REAL_COUNT : PFFFT_WIDE_ZEN4_COMPLEX_COUNT;
    return real ? pffft_wide_zen4_real_sizes : pffft_wide_zen4_complex_sizes;
  }
  *count = 0;
  return NULL;
}

static void test_select_rule(void) {
  enum { O = PFFFT_X86_WIDE_OFF, I = PFFFT_X86_WIDE_INFERRED, M = PFFFT_X86_WIDE_ZEN3,
         Z4 = PFFFT_X86_WIDE_ZEN4, ZS = PFFFT_X86_WIDE_ZEN3_SERVER, ZD = PFFFT_X86_WIDE_ZEN3_DESKTOP };
  int t, i, measured_only = 0, inferred_checked = 0, limited = 0, zen4_differs = 0;
  int desktop_only = 0, server_only = 0;
  for (t = 0; t < 2; ++t) {
    const pffft_transform_t type = t ? PFFFT_REAL : PFFFT_COMPLEX;
    int mc, zc, dc, sc;
    const int *mt = measured_table(M, t, &mc), *zt = measured_table(Z4, t, &zc);
    const int *dt = measured_table(ZD, t, &dc), *st = measured_table(ZS, t, &sc);
    const int *it = t ? pffft_wide_inferred_real_sizes : pffft_wide_inferred_complex_sizes;
    const int ic = t ? PFFFT_WIDE_INFERRED_REAL_COUNT : PFFFT_WIDE_INFERRED_COMPLEX_COUNT;
    int N;
    for (i = 0; i < zc; ++i) {
      check(wide_eligible(zt[i], type), "select: Zen 4 table entry the backend cannot handle", zt[i], t);
      if (i) check(zt[i] > zt[i-1], "select: Zen 4 table not increasing", zt[i], t);
    }
    for (i = 0; i < dc; ++i) {
      check(wide_eligible(dt[i], type), "select: Zen 3 desktop entry the backend cannot handle", dt[i], t);
      if (i) check(dt[i] > dt[i-1], "select: Zen 3 desktop table not increasing", dt[i], t);
    }
    for (i = 0; i < sc; ++i) {
      check(wide_eligible(st[i], type), "select: Zen 3 server entry the backend cannot handle", st[i], t);
      if (i) check(st[i] > st[i-1], "select: Zen 3 server table not increasing", st[i], t);
    }
    /* the sizes for unmeasured Zen 3 models passed on both measured ones */
    for (i = 0; i < mc; ++i)
      check(in_table(dt, dc, mt[i]) && in_table(st, sc, mt[i]), "select: Zen 3 size not in both models' tables",
            mt[i], t);
    for (i = 0; i < ic; ++i) {
      check(in_table(mt, mc, it[i]), "select: inferred size not in the Zen 3 table", it[i], t);
      check(in_table(zt, zc, it[i]), "select: inferred size not in the Zen 4 table", it[i], t);
      if (i) check(it[i] > it[i-1], "select: inferred table not increasing", it[i], t);
    }
    for (N = 64; N <= (1 << 20); N += 64) {
      const unsigned long long ws = pffft_wide_working_set(N, type);
      const int in_m = in_table(mt, mc, N), in_i = in_table(it, ic, N);
      check(pffft_wide_supports(N, type) == wide_eligible(N, type), "select: supported sizes", N, t);
      if (!wide_eligible(N, type)) continue;
      check(pffft_wide_default(N, type, M, 0) == in_m, "select: measured", N, t);
      check(pffft_wide_default(N, type, Z4, 0) == in_table(zt, zc, N), "select: Zen 4", N, t);
      check(pffft_wide_default(N, type, ZD, 0) == in_table(dt, dc, N), "select: Zen 3 desktop", N, t);
      check(pffft_wide_default(N, type, ZS, 0) == in_table(st, sc, N), "select: Zen 3 server", N, t);
      if (in_table(dt, dc, N) && !in_table(st, sc, N)) ++desktop_only;
      if (in_table(st, sc, N) && !in_table(dt, dc, N)) ++server_only;
      if (in_table(zt, zc, N) != in_m) ++zen4_differs;
      check(pffft_wide_default(N, type, O, 1ull << 40) == 0, "select: off", N, t);
      check(pffft_wide_default(N, type, I, 0) == 0, "select: inferred, L2 unknown", N, t);
      check(pffft_wide_default(N, type, I, 1ull << 40) == in_i, "select: inferred, large L2", N, t);
      check(pffft_wide_default(N, type, I, ws) == in_i, "select: inferred, L2 = working set", N, t);
      check(pffft_wide_default(N, type, I, ws - 1) == 0, "select: inferred, L2 below working set", N, t);
      check(ws == 4ull * (unsigned long long)N * (t ? 4u : 8u), "select: working set", N, t);
      if (in_m && !in_i) ++measured_only;
      if (in_i) ++inferred_checked;
      if (in_i && pffft_wide_default(N, type, I, 256u * 1024) == 0) ++limited;
    }
  }
  /* the cases that tell the tables and the limit apart exist */
  check(measured_only > 0 && inferred_checked > 0 && limited > 0 && zen4_differs > 0,
        "select: distinguishing cases", measured_only, limited);
  check(desktop_only > 0 && server_only > 0, "select: Zen 3 models' tables differ", desktop_only, server_only);
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


/* The default uses 8 lanes only at table sizes, and either at all of them
   (the CPU is one of the measured ones) or at none. */
/* this CPU's class and L2 size as pffft.c derives them, with the build
   options the library was compiled with (tests/CMakeLists.txt passes them) */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_ARM64EC)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
static void real_cpuid(void *ctx, unsigned leaf, unsigned subleaf, unsigned *r) {
  (void)ctx;
#  if defined(_MSC_VER)
  {
    int v[4];
    __cpuidex(v, (int)leaf, (int)subleaf);
    r[0] = (unsigned)v[0]; r[1] = (unsigned)v[1]; r[2] = (unsigned)v[2]; r[3] = (unsigned)v[3];
  }
#  else
  __cpuid_count(leaf, subleaf, r[0], r[1], r[2], r[3]);
#  endif
}
static unsigned long long real_xgetbv(void *ctx, unsigned index) {
  (void)ctx; (void)index;
  return 0x7;   /* only used when the 8-lane backend is available, i.e. the OS has YMM state */
}
static int this_cpu_class(unsigned long long *l2) {
  pffft_x86_cpu_ops ops;
  pffft_x86_cpu_id id;
  int c;
  ops.cpuid = real_cpuid; ops.xgetbv = real_xgetbv; ops.ctx = NULL;
  *l2 = 0;
  pffft_x86_probe_wide(&ops, 0, &id);
  c = pffft_x86_wide_class(&id);
  /* "raw class" is what tests/run_sde_checks.sh compares with the emulated chip */
  printf("CPU vendor %d family %02Xh model %02Xh hybrid %d, raw class %s\n", id.vendor, id.family, id.model,
         id.hybrid, c == PFFFT_X86_WIDE_ZEN3 ? "measured (zen3)" : c == PFFFT_X86_WIDE_ZEN4 ? "measured (zen4)"
         : c == PFFFT_X86_WIDE_ZEN3_DESKTOP ? "measured (zen3 desktop)"
         : c == PFFFT_X86_WIDE_ZEN3_SERVER ? "measured (zen3 server)"
         : c == PFFFT_X86_WIDE_INFERRED ? "inferred" : "off");
#  if defined(PFFFT_WIDE_UNKNOWN_CPUS)
  return PFFFT_X86_WIDE_ZEN3;
#  else
#    if !defined(PFFFT_WIDE_INFERRED)
  if (c == PFFFT_X86_WIDE_INFERRED) c = PFFFT_X86_WIDE_OFF;
#    endif
  if (c == PFFFT_X86_WIDE_INFERRED) *l2 = pffft_x86_l2_bytes(&ops, &id);
  return c;
#  endif
}
#else
static int this_cpu_class(unsigned long long *l2) { *l2 = 0; return PFFFT_X86_WIDE_OFF; }
#endif

static void test_size_table(int have_wide) {
  int t, i, N;
  unsigned long long l2;
  const int cpu_class = this_cpu_class(&l2);
  printf("CPU class %s, L2 %llu KiB\n", PFFFT_X86_WIDE_IS_MEASURED(cpu_class) ? "measured"
         : cpu_class == PFFFT_X86_WIDE_INFERRED ? "inferred" : "off", l2 >> 10);
  for (t = 0; t < 2; ++t) {
    const pffft_transform_t type = t ? PFFFT_REAL : PFFFT_COMPLEX;
    /* the table this CPU uses (the Zen 3 one on other CPUs, for the checks below) */
    int count;
    const int *table = measured_table(PFFFT_X86_WIDE_IS_MEASURED(cpu_class) ? cpu_class : PFFFT_X86_WIDE_ZEN3,
                                      t, &count);
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
#if defined(PFFFT_WIDE_ALL_SIZES)
      check(lanes == (have_wide ? 8 : pffft_simd_size()), "size table: all sizes", N, lanes);
#else
      /* exactly the rule of pffft_wide_select.h for this CPU */
      check(lanes == (have_wide && pffft_wide_default(N, type, cpu_class, l2) ? 8 : pffft_simd_size()),
            "size table: default lanes", N, lanes);
      if (!in_table(table, count, N)) check(lanes == pffft_simd_size(), "size table: 8 lanes at a size not in the table", N, lanes);
#endif
    }
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

/* the eight transform modes (ops 0..7), then the spectral operations */
#define NUM_TRANSFORM_OPS 8
enum { OP_REORDER_FWD = NUM_TRANSFORM_OPS, OP_REORDER_BWD, OP_ZCONV, OP_ZCONV_SCALE,
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
  /* CI jobs on AVX2+FMA machines with builds that contain the backend set
     this (CMake option PFFFT_TEST_REQUIRE_WIDE), so that a build that left
     it out cannot pass by skipping */
  if (!have_wide && getenv("PFFFT_TEST_REQUIRE_WIDE") && strcmp(getenv("PFFFT_TEST_REQUIRE_WIDE"), "1") == 0) {
    printf("FAIL: the 8-lane backend is required (PFFFT_TEST_REQUIRE_WIDE=1)\n");
    return 1;
  }

  test_decode();
  test_probe();
  test_classes();
  test_l2();
  test_select_rule();
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
