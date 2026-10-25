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
// expect: none
	.text
base_caller2:
	call w8_outer2
	ret
w8_outer2:
	vaddps %ymm0, %ymm0, %ymm0
	call w8_inner2
	vzeroupper
	addss %xmm1, %xmm1
	ret
w8_inner2:
	vmulps %ymm0, %ymm0, %ymm0
	ret
