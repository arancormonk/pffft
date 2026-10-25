/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Runtime detection for the 8-lane AVX2 float backend (pffft_wide_impl.h).

  The CPUID and XGETBV instructions are reached through a small table of
  function pointers so that tests can replace them with fakes and check
  both the decoding and the order of the probes: XGETBV may only run once
  CPUID reports OSXSAVE, and leaf 7 may only be queried when CPUID leaf 0
  reports it. Everything is static; pffft.c and tests include this file.
  The probe keeps no state: pffft_new_setup() runs it on every call.
*/

#ifndef PFFFT_CPU_X86_H
#define PFFFT_CPU_X86_H

#define PFFFT_X86_VENDOR_OTHER 0
#define PFFFT_X86_VENDOR_INTEL 1
#define PFFFT_X86_VENDOR_AMD   2

#define PFFFT_X86_ECX1_FMA     (1u << 12)
#define PFFFT_X86_ECX1_OSXSAVE (1u << 27)
#define PFFFT_X86_ECX1_AVX     (1u << 28)
#define PFFFT_X86_EBX7_AVX2    (1u << 5)
#define PFFFT_X86_XCR0_SSE_AVX 0x6u  /* XMM and YMM state enabled by the OS */

typedef struct {
  /* regs[0..3] = eax, ebx, ecx, edx of CPUID(leaf, subleaf) */
  void (*cpuid)(void *ctx, unsigned leaf, unsigned subleaf, unsigned *regs);
  /* XCR register 'index' */
  unsigned long long (*xgetbv)(void *ctx, unsigned index);
  void *ctx;
} pffft_x86_cpu_ops;

typedef struct {
  int vendor;
  unsigned family, model;
} pffft_x86_cpu_id;

/* pure decision from the register values */
static int pffft_x86_decode_wide(unsigned max_leaf, unsigned ecx1, unsigned ebx7,
                                 unsigned long long xcr0, int need_fma) {
  if (max_leaf < 7) return 0;
  if (!(ecx1 & PFFFT_X86_ECX1_OSXSAVE)) return 0;
  if (!(ecx1 & PFFFT_X86_ECX1_AVX)) return 0;
  if (need_fma && !(ecx1 & PFFFT_X86_ECX1_FMA)) return 0;
  if ((xcr0 & PFFFT_X86_XCR0_SSE_AVX) != PFFFT_X86_XCR0_SSE_AVX) return 0;
  if (!(ebx7 & PFFFT_X86_EBX7_AVX2)) return 0;
  return 1;
}

/* nonzero when the CPU and OS can run AVX2 (and FMA, if need_fma) code;
   fills *id with the vendor and display family/model when non-NULL */
static int pffft_x86_probe_wide(const pffft_x86_cpu_ops *ops, int need_fma, pffft_x86_cpu_id *id) {
  unsigned r[4], max_leaf, ecx1, ebx7;
  unsigned long long xcr0;
  if (id) { id->vendor = PFFFT_X86_VENDOR_OTHER; id->family = id->model = 0; }

  ops->cpuid(ops->ctx, 0, 0, r);
  max_leaf = r[0];
  if (max_leaf < 7) return 0;
  if (id) {
    /* vendor string in ebx, edx, ecx */
    if (r[1] == 0x756e6547u && r[3] == 0x49656e69u && r[2] == 0x6c65746eu)
      id->vendor = PFFFT_X86_VENDOR_INTEL;       /* "GenuineIntel" */
    else if (r[1] == 0x68747541u && r[3] == 0x69746e65u && r[2] == 0x444d4163u)
      id->vendor = PFFFT_X86_VENDOR_AMD;         /* "AuthenticAMD" */
  }

  ops->cpuid(ops->ctx, 1, 0, r);
  ecx1 = r[2];
  if (id) {
    unsigned family = (r[0] >> 8) & 0xFu, model = (r[0] >> 4) & 0xFu;
    if (family == 0xFu) family += (r[0] >> 20) & 0xFFu;
    if (family == 0x6u || family >= 0xFu) model += ((r[0] >> 16) & 0xFu) << 4;
    id->family = family;
    id->model = model;
  }
  if (!(ecx1 & PFFFT_X86_ECX1_OSXSAVE) || !(ecx1 & PFFFT_X86_ECX1_AVX)) return 0;
  if (need_fma && !(ecx1 & PFFFT_X86_ECX1_FMA)) return 0;

  xcr0 = ops->xgetbv(ops->ctx, 0);
  ops->cpuid(ops->ctx, 7, 0, r);
  ebx7 = r[1];
  return pffft_x86_decode_wide(max_leaf, ecx1, ebx7, xcr0, need_fma);
}

/*
  CPUs on which the 8-lane backend was measured to be faster than the
  4-lane one for every mode of the enabled sizes (see
  benchmarks/perf_analysis/README.md). On other CPUs the 4-lane backend
  stays the default: ISA support alone does not tell how fast 256-bit
  arithmetic, lane-crossing permutes or the caches are.
*/
static int pffft_x86_wide_measured(const pffft_x86_cpu_id *id) {
  if (id->vendor == PFFFT_X86_VENDOR_AMD && id->family == 0x19u) {
    /* Zen 3: Milan/Chagall, Vermeer, Rembrandt/Cezanne */
    if (id->model <= 0x0Fu) return 1;
    if (id->model >= 0x20u && id->model <= 0x2Fu) return 1;
    if (id->model >= 0x40u && id->model <= 0x5Fu) return 1;
  }
  return 0;
}

#endif /* PFFFT_CPU_X86_H */
