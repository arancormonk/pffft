/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Runtime detection for the 8-lane AVX2 float backend (pffft_wide_impl.h).

  The CPUID and XGETBV instructions are reached through a small table of
  function pointers so that tests can replace them with fakes and check
  both the decoding and the order of the probes: XGETBV may only run once
  CPUID reports OSXSAVE, and a leaf may only be queried when CPUID leaf 0
  (or 8000_0000h for the extended leaves) reports it. Everything is static;
  pffft.c and tests include this file. The probe keeps no state:
  pffft_new_setup() runs it on every call.
*/

#ifndef PFFFT_CPU_X86_H
#define PFFFT_CPU_X86_H

/* MSVC with the 8-lane backend in pffft_wide_tu.c (see pffft.c) */
#if defined(PFFFT_WIDE_TU) && defined(_MSC_VER) && !defined(__clang__) && !defined(PFFFT_WIDE_SEPARATE)
#  define PFFFT_WIDE_SEPARATE 1
#endif

#define PFFFT_X86_VENDOR_OTHER 0
#define PFFFT_X86_VENDOR_INTEL 1
#define PFFFT_X86_VENDOR_AMD   2

#define PFFFT_X86_ECX1_SSE3    (1u << 0)
#define PFFFT_X86_ECX1_SSSE3   (1u << 9)
#define PFFFT_X86_ECX1_FMA     (1u << 12)
#define PFFFT_X86_ECX1_SSE41   (1u << 19)
#define PFFFT_X86_ECX1_SSE42   (1u << 20)
#define PFFFT_X86_ECX1_POPCNT  (1u << 23)
#define PFFFT_X86_ECX1_XSAVE   (1u << 26)
#define PFFFT_X86_ECX1_OSXSAVE (1u << 27)
#define PFFFT_X86_ECX1_AVX     (1u << 28)
#define PFFFT_X86_ECX1_MOVBE   (1u << 22)
#define PFFFT_X86_ECX1_F16C    (1u << 29)
#define PFFFT_X86_EBX7_BMI1    (1u << 3)
#define PFFFT_X86_EBX7_AVX2    (1u << 5)
#define PFFFT_X86_EBX7_BMI2    (1u << 8)
#define PFFFT_X86_ECX81_LZCNT  (1u << 5)    /* leaf 8000_0001h */
#define PFFFT_X86_EDX7_HYBRID  (1u << 15)
#define PFFFT_X86_ECX81_TOPOEXT (1u << 22)  /* leaf 8000_0001h: AMD leaf 8000_001Dh exists */
#define PFFFT_X86_XCR0_SSE_AVX 0x6u  /* XMM and YMM state enabled by the OS */

/*
  The instruction set extensions the 8-lane code may use: everything the
  compilers' avx2,fma target implies (GCC and Clang also enable SSE3 to
  SSE4.2, POPCNT and XSAVE with it), and when MSVC compiles it with
  /arch:AVX2 (PFFFT_WIDE_SEPARATE) the rest of x86-64-v3, since /arch:AVX2
  may also use BMI and related instructions. pffft_x86_decode_wide()
  requires every entry (fma only when the build uses FMA), and
  tests/check_isa_baseline.py reads this table for the instructions it
  allows in pffft_w8_* functions (the _TU entries with --wide-tu).
  Entries: name, CPUID register (1: leaf 1 ecx, 7: leaf 7 ebx, 0x81: leaf
  8000_0001h ecx), bit.
*/
#define PFFFT_X86_WIDE_ISA_ENTRY(name, reg, bit) { name, reg, bit }
#define PFFFT_X86_WIDE_ISA_TU_ENTRY(name, reg, bit) { name, reg, bit }
typedef struct { const char *name; int reg; unsigned bit; } pffft_x86_isa_entry;
static const pffft_x86_isa_entry pffft_x86_wide_isa[] = {
  PFFFT_X86_WIDE_ISA_ENTRY("sse3",   1, PFFFT_X86_ECX1_SSE3),
  PFFFT_X86_WIDE_ISA_ENTRY("ssse3",  1, PFFFT_X86_ECX1_SSSE3),
  PFFFT_X86_WIDE_ISA_ENTRY("sse4.1", 1, PFFFT_X86_ECX1_SSE41),
  PFFFT_X86_WIDE_ISA_ENTRY("sse4.2", 1, PFFFT_X86_ECX1_SSE42),
  PFFFT_X86_WIDE_ISA_ENTRY("popcnt", 1, PFFFT_X86_ECX1_POPCNT),
  PFFFT_X86_WIDE_ISA_ENTRY("xsave",  1, PFFFT_X86_ECX1_XSAVE),
  PFFFT_X86_WIDE_ISA_ENTRY("avx",    1, PFFFT_X86_ECX1_AVX),
  PFFFT_X86_WIDE_ISA_ENTRY("avx2",   7, PFFFT_X86_EBX7_AVX2),
  PFFFT_X86_WIDE_ISA_ENTRY("fma",    1, PFFFT_X86_ECX1_FMA)
#if defined(PFFFT_WIDE_SEPARATE)
  ,
  PFFFT_X86_WIDE_ISA_TU_ENTRY("bmi1",  7, PFFFT_X86_EBX7_BMI1),
  PFFFT_X86_WIDE_ISA_TU_ENTRY("bmi2",  7, PFFFT_X86_EBX7_BMI2),
  PFFFT_X86_WIDE_ISA_TU_ENTRY("lzcnt", 0x81, PFFFT_X86_ECX81_LZCNT),
  PFFFT_X86_WIDE_ISA_TU_ENTRY("movbe", 1, PFFFT_X86_ECX1_MOVBE),
  PFFFT_X86_WIDE_ISA_TU_ENTRY("f16c",  1, PFFFT_X86_ECX1_F16C)
#endif
};
#define PFFFT_X86_WIDE_ISA_COUNT ((int)(sizeof(pffft_x86_wide_isa) / sizeof(pffft_x86_wide_isa[0])))

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
  int hybrid;                 /* CPUID.(EAX=7,ECX=0):EDX[15] */
  unsigned max_leaf;
} pffft_x86_cpu_id;

/* pure decision from the register values */
static int pffft_x86_decode_wide(unsigned max_leaf, unsigned ecx1, unsigned ebx7, unsigned ecx81,
                                 unsigned long long xcr0, int need_fma) {
  int k;
  if (max_leaf < 7) return 0;
  if (!(ecx1 & PFFFT_X86_ECX1_OSXSAVE)) return 0;
  if ((xcr0 & PFFFT_X86_XCR0_SSE_AVX) != PFFFT_X86_XCR0_SSE_AVX) return 0;
  for (k = 0; k < PFFFT_X86_WIDE_ISA_COUNT; ++k) {
    const pffft_x86_isa_entry *e = &pffft_x86_wide_isa[k];
    if (!need_fma && e->bit == PFFFT_X86_ECX1_FMA && e->reg == 1) continue;
    if (!((e->reg == 7 ? ebx7 : e->reg == 0x81 ? ecx81 : ecx1) & e->bit)) return 0;
  }
  return 1;
}

/* nonzero when the CPU and OS can run AVX2 (and FMA, if need_fma) code;
   fills *id with the vendor and display family/model when non-NULL */
static int pffft_x86_probe_wide(const pffft_x86_cpu_ops *ops, int need_fma, pffft_x86_cpu_id *id) {
  unsigned r[4], max_leaf, ecx1, ebx7, ecx81 = 0;
  unsigned long long xcr0;
  if (id) {
    id->vendor = PFFFT_X86_VENDOR_OTHER; id->family = id->model = 0;
    id->hybrid = 0; id->max_leaf = 0;
  }

  ops->cpuid(ops->ctx, 0, 0, r);
  max_leaf = r[0];
  if (id) id->max_leaf = max_leaf;
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
  if (id) id->hybrid = (r[3] & PFFFT_X86_EDX7_HYBRID) != 0;
  ops->cpuid(ops->ctx, 0x80000000u, 0, r);
  if (r[0] >= 0x80000001u && r[0] <= 0x8000FFFFu) {
    ops->cpuid(ops->ctx, 0x80000001u, 0, r);
    ecx81 = r[2];
  }
  return pffft_x86_decode_wide(max_leaf, ecx1, ebx7, ecx81, xcr0, need_fma);
}

/* size in bytes of a deterministic cache parameter leaf's cache (Intel leaf
   4, AMD leaf 8000_001Dh: same layout, all four counts stored minus one) */
static unsigned long long pffft_x86_cache_bytes(const unsigned *r) {
  const unsigned long long ways = ((r[1] >> 22) & 0x3FFu) + 1;
  const unsigned long long partitions = ((r[1] >> 12) & 0x3FFu) + 1;
  const unsigned long long line = (r[1] & 0xFFFu) + 1;
  const unsigned long long sets = (unsigned long long)r[2] + 1;
  return ways * partitions * line * sets;   /* at most 2^52: no overflow */
}

/* the level-2 data or unified cache in a deterministic cache leaf, 0 if none */
static unsigned long long pffft_x86_l2_from_leaf(const pffft_x86_cpu_ops *ops, unsigned leaf) {
  unsigned r[4], k;
  for (k = 0; k < 32; ++k) {
    unsigned type, level;
    ops->cpuid(ops->ctx, leaf, k, r);
    type = r[0] & 0x1Fu;
    level = (r[0] >> 5) & 0x7u;
    if (type == 0) break;                   /* no more caches */
    if (level == 2 && (type == 1 || type == 3)) return pffft_x86_cache_bytes(r);
  }
  return 0;
}

/*
  Per-core L2 size in bytes from the deterministic cache leaves (Intel leaf
  4; AMD leaf 8000_001Dh when TOPOEXT is set), else AMD's legacy leaf
  8000_0006h ECX[31:16] in KiB; 0 when unknown. Intel's legacy leaf is not
  used: it can disagree with leaf 4 (e.g. 256 KiB vs 512 KiB on Ice Lake).
  Needs the vendor and max_leaf from pffft_x86_probe_wide().
*/
static unsigned long long pffft_x86_l2_bytes(const pffft_x86_cpu_ops *ops, const pffft_x86_cpu_id *id) {
  unsigned r[4], max_ext;
  if (id->vendor == PFFFT_X86_VENDOR_INTEL) {
    return id->max_leaf >= 4 ? pffft_x86_l2_from_leaf(ops, 4) : 0;
  }
  if (id->vendor != PFFFT_X86_VENDOR_AMD) return 0;
  ops->cpuid(ops->ctx, 0x80000000u, 0, r);
  max_ext = r[0];
  if (max_ext < 0x80000001u || max_ext > 0x8000FFFFu) return 0;
  ops->cpuid(ops->ctx, 0x80000001u, 0, r);
  if ((r[2] & PFFFT_X86_ECX81_TOPOEXT) && max_ext >= 0x8000001Du) {
    unsigned long long l2 = pffft_x86_l2_from_leaf(ops, 0x8000001Du);
    if (l2) return l2;
  }
  if (max_ext >= 0x80000006u) {
    ops->cpuid(ops->ctx, 0x80000006u, 0, r);
    return (unsigned long long)(r[2] >> 16) * 1024u;
  }
  return 0;
}

/*
  Which CPUs use the 8-lane backend by default (see
  benchmarks/perf_analysis/README.md, "CPU classes"):

  measured  the backend was measured faster than the 4-lane one for every
            mode at the sizes of the class's own table, in each sweep
            behind it: Zen 3 desktop (Vermeer: Ryzen 9 5950X), Zen 3 server
            (Genesis, Chagall: EPYC 7763 on GitHub-hosted runners), Zen 4
            (EPYC 9V74 on GitHub-hosted runners). The Zen 3 models that
            were not measured (the Cezanne and Rembrandt APUs) use the
            sizes that passed on both measured Zen 3 models.
  inferred  not measured; cores whose documented 256-bit FP datapaths are
            full width and that have no AVX frequency licences, so the
            per-element instruction saving should carry over: the sizes of
            the stricter inferred table, within an L2 working-set limit.
  off       everything else: 4 lanes. ISA support alone does not tell how
            fast 256-bit arithmetic, lane-crossing permutes or the caches
            are; this includes 128-bit datapaths (Zen, Zen+, Hygon,
            Excavator, Atom cores), hybrid parts (a thread may run on its
            E-cores), Intel server parts, unknown models and future
            families.

  Family/model ranges from LLVM llvm/lib/TargetParser/Host.cpp (main, 2026),
  cross-checked with AMD's family/model table and Linux intel-family.h.
*/
#define PFFFT_X86_WIDE_OFF      0
#define PFFFT_X86_WIDE_INFERRED 1
#define PFFFT_X86_WIDE_ZEN3     2   /* measured microarchitecture: both Zen 3 models' sizes */
#define PFFFT_X86_WIDE_ZEN4     3   /* measured */
#define PFFFT_X86_WIDE_ZEN3_DESKTOP 4   /* measured */
#define PFFFT_X86_WIDE_ZEN3_SERVER  5   /* measured */
#define PFFFT_X86_WIDE_IS_MEASURED(c) ((c) >= PFFFT_X86_WIDE_ZEN3)

typedef struct { unsigned family, lo, hi; } pffft_x86_model_range;

static const pffft_x86_model_range pffft_x86_amd_zen3_server[] = {
  { 0x19, 0x00, 0x0F }    /* Zen 3: Genesis, Chagall */
};
static const pffft_x86_model_range pffft_x86_amd_zen3_desktop[] = {
  { 0x19, 0x20, 0x2F }    /* Zen 3: Vermeer */
};
static const pffft_x86_model_range pffft_x86_amd_zen3[] = {
  { 0x19, 0x40, 0x5F }    /* Zen 3+: Rembrandt; Zen 3: Cezanne */
};
static const pffft_x86_model_range pffft_x86_amd_zen4[] = {
  { 0x19, 0x10, 0x1F },   /* Zen 4: Stones, Storm Peak */
  { 0x19, 0x60, 0x7F },   /* Zen 4: Raphael, Phoenix, Phoenix 2, Hawk Point */
  { 0x19, 0xA0, 0xAF }    /* Zen 4: Stones-Dense */
};
static const pffft_x86_model_range pffft_x86_amd_inferred[] = {
  { 0x17, 0x30, 0x3F },   /* Zen 2: Starship (Rome) */
  { 0x17, 0x47, 0x47 },   /* Zen 2: Cardinal */
  { 0x17, 0x60, 0x7F },   /* Zen 2: Renoir, Lucienne, Matisse */
  { 0x17, 0x84, 0x87 },   /* Zen 2: ProjectX */
  { 0x17, 0x90, 0xAF },   /* Zen 2: VanGogh, Mero, Mendocino */
  { 0x19, 0x30, 0x3F },   /* Zen 3: Badami */
  { 0x1A, 0x00, 0x4F },   /* Zen 5: Breithorn(-Dense), Strix 1-3, Granite Ridge */
  { 0x1A, 0x60, 0x77 },   /* Zen 5: Krackan 1, Sarlak */
  { 0x1A, 0xD0, 0xD7 }    /* Zen 5: Annapurna */
};
/* family 6 client big cores, not hybrid */
static const unsigned char pffft_x86_intel_inferred[] = {
  0x3C, 0x45, 0x46,                     /* Haswell */
  0x3D, 0x47,                           /* Broadwell */
  0x4E, 0x5E, 0x8E, 0x9E, 0xA5, 0xA6,   /* Skylake, Kaby/Coffee/Comet Lake */
  0x66,                                 /* Cannon Lake */
  0x7D, 0x7E,                           /* Ice Lake client */
  0x8C, 0x8D,                           /* Tiger Lake */
  0xA7                                  /* Rocket Lake */
};
/* family 6 hybrid-generation models with P-core-only parts: inferred when
   the Hybrid bit is clear (e.g. i5-12400, i3-13100); off otherwise */
static const unsigned char pffft_x86_intel_pcore_only[] = {
  0x97,                                 /* Alder Lake-S */
  0xB7, 0xBF                            /* Raptor Lake-S */
};

static int pffft_x86_in_ranges(const pffft_x86_model_range *r, int n, const pffft_x86_cpu_id *id) {
  int k;
  for (k = 0; k < n; ++k)
    if (id->family == r[k].family && id->model >= r[k].lo && id->model <= r[k].hi) return 1;
  return 0;
}

static int pffft_x86_in_models(const unsigned char *m, int n, unsigned model) {
  int k;
  for (k = 0; k < n; ++k) if (model == m[k]) return 1;
  return 0;
}

static int pffft_x86_wide_class(const pffft_x86_cpu_id *id) {
#define PFFFT_X86_COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))
  if (id->vendor == PFFFT_X86_VENDOR_AMD) {
    if (pffft_x86_in_ranges(pffft_x86_amd_zen3_desktop, PFFFT_X86_COUNT(pffft_x86_amd_zen3_desktop), id))
      return PFFFT_X86_WIDE_ZEN3_DESKTOP;
    if (pffft_x86_in_ranges(pffft_x86_amd_zen3_server, PFFFT_X86_COUNT(pffft_x86_amd_zen3_server), id))
      return PFFFT_X86_WIDE_ZEN3_SERVER;
    if (pffft_x86_in_ranges(pffft_x86_amd_zen3, PFFFT_X86_COUNT(pffft_x86_amd_zen3), id))
      return PFFFT_X86_WIDE_ZEN3;
    if (pffft_x86_in_ranges(pffft_x86_amd_zen4, PFFFT_X86_COUNT(pffft_x86_amd_zen4), id))
      return PFFFT_X86_WIDE_ZEN4;
    if (pffft_x86_in_ranges(pffft_x86_amd_inferred, PFFFT_X86_COUNT(pffft_x86_amd_inferred), id))
      return PFFFT_X86_WIDE_INFERRED;
  } else if (id->vendor == PFFFT_X86_VENDOR_INTEL && id->family == 6 && !id->hybrid) {
    if (pffft_x86_in_models(pffft_x86_intel_inferred, PFFFT_X86_COUNT(pffft_x86_intel_inferred), id->model)
        || pffft_x86_in_models(pffft_x86_intel_pcore_only, PFFFT_X86_COUNT(pffft_x86_intel_pcore_only), id->model))
      return PFFFT_X86_WIDE_INFERRED;
  }
#undef PFFFT_X86_COUNT
  return PFFFT_X86_WIDE_OFF;
}

#endif /* PFFFT_CPU_X86_H */
