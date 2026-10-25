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
// expect: transition w8_inner
// expect: transition w8_inner
	.text
base_caller:
	call w8_outer
	ret
w8_outer:
	vaddps %ymm0, %ymm0, %ymm0
	call w8_inner
	vzeroupper
	ret
w8_inner:
	sub $0x18, %rsp
	movaps %xmm6, (%rsp)
	movaps (%rsp), %xmm6
	add $0x18, %rsp
	ret
