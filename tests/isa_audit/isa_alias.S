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
// expect: isa base_alias
// expect: isa base_alias
// one function under a baseline and an 8-lane name (identical functions
// folded by the linker): baseline code can run it, so no AVX
	.text
	.globl base_alias
	.globl w8_alias
#if !defined(_WIN32)
	.type base_alias, @function
	.type w8_alias, @function
#endif
base_alias:
w8_alias:
	vaddps %ymm0, %ymm0, %ymm0
	vzeroupper
	ret
