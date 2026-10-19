/* Configure-time probe for MSVC-style compilers, which have no /arch:native.
 * Prints the highest /arch level the build host can run: AVX512, AVX2, AVX or
 * none. The checks mirror what MSVC may emit for each level: AVX2 also covers
 * FMA, F16C, BMI1/2, LZCNT and MOVBE, and every level needs the OS to save the
 * corresponding register state (XCR0). */
#include <stdio.h>

#if (defined(_M_X64) || defined(_M_IX86)) && !defined(_M_ARM64EC)
#include <intrin.h>

static void cpuid(int leaf, int sub, unsigned regs[4]) {
    int r[4];
    __cpuidex(r, leaf, sub);
    regs[0] = (unsigned)r[0];
    regs[1] = (unsigned)r[1];
    regs[2] = (unsigned)r[2];
    regs[3] = (unsigned)r[3];
}

static unsigned long long xcr0(void) {
#if defined(__clang__)
    /* clang-cl's _xgetbv needs the xsave target feature; emit XGETBV directly */
    unsigned lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#else
    return _xgetbv(0);
#endif
}

static const char* host_arch(void) {
    unsigned l0[4], l1[4], l7[4] = {0, 0, 0, 0}, e1[4] = {0, 0, 0, 0}, emax[4];
    unsigned long long xcr;
    int v3;

    cpuid(0, 0, l0);
    if (l0[0] < 1) {
        return "none";
    }
    cpuid(1, 0, l1);
    /* OSXSAVE (27) and AVX (28) */
    if ((l1[2] & (1u << 27)) == 0 || (l1[2] & (1u << 28)) == 0) {
        return "none";
    }
    xcr = xcr0();
    if ((xcr & 0x6) != 0x6) { /* XMM and YMM state */
        return "none";
    }
    if (l0[0] >= 7) {
        cpuid(7, 0, l7);
    }
    cpuid((int)0x80000000u, 0, emax);
    if (emax[0] >= 0x80000001u) {
        cpuid((int)0x80000001u, 0, e1);
    }
    v3 = (l1[2] & (1u << 12)) != 0       /* FMA */
         && (l1[2] & (1u << 22)) != 0    /* MOVBE */
         && (l1[2] & (1u << 29)) != 0    /* F16C */
         && (l7[1] & (1u << 3)) != 0     /* BMI1 */
         && (l7[1] & (1u << 5)) != 0     /* AVX2 */
         && (l7[1] & (1u << 8)) != 0     /* BMI2 */
         && (e1[2] & (1u << 5)) != 0;    /* LZCNT */
    if (!v3) {
        return "AVX";
    }
    if ((l7[1] & (1u << 16)) != 0        /* AVX512F */
        && (l7[1] & (1u << 17)) != 0     /* AVX512DQ */
        && (l7[1] & (1u << 28)) != 0     /* AVX512CD */
        && (l7[1] & (1u << 30)) != 0     /* AVX512BW */
        && (l7[1] & (1u << 31)) != 0     /* AVX512VL */
        && (xcr & 0xE6) == 0xE6) {       /* opmask and ZMM state */
        return "AVX512";
    }
    return "AVX2";
}
#else
static const char* host_arch(void) { return "none"; }
#endif

int main(void) {
    printf("PFFFT_HOST_ARCH=%s\n", host_arch());
    return 0;
}
