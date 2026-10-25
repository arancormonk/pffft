// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
// Fixture for check_isa_baseline.py --self-test; "expect:" lines list the
// findings (rule, function) the audit must report, one per finding.
#if defined(_WIN32)
#  define RODATA16 .section .rdata$a16,"dr"
#  define RODATA32 .section .rdata$a32,"dr"
#else
#  define RODATA16 .section .rodata.a16,"a"
#  define RODATA32 .section .rodata.a32,"a"
#endif
// expect: isa base_cd
// expect: isa base_cd
// two code sections of the same name (MSVC's COMDAT .text$mn): each
// function must keep its own name, so the AVX in base_cd is found
#if defined(_WIN32) && defined(__clang__)
	.section .text$mn,"xr",discard,base_cd
#elif defined(_WIN32)
	.section .text$mn,"xr"
	.linkonce discard
#else
	.section .text.cd1,"ax",@progbits
#endif
	.globl base_cd
base_cd:
	vaddps %ymm0, %ymm0, %ymm0
	vzeroupper
	ret
#if defined(_WIN32) && defined(__clang__)
	.section .text$mn,"xr",discard,w8_cd
#elif defined(_WIN32)
	.section .text$mn,"xr"
	.linkonce discard
#else
	.section .text.cd2,"ax",@progbits
#endif
	.globl w8_cd
w8_cd:
	vaddps %ymm0, %ymm0, %ymm0
	vzeroupper
	ret
