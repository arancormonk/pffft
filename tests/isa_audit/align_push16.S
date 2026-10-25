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
// expect: align w8_p16
// expect: align w8_pf16
// 16-bit pushes and pops move rsp by 2, not 8: the stack slot is not known
// to be aligned after them
	.text
	.globl w8_p16
w8_p16:
	push %ax
	vmovaps (%rsp), %xmm0
	pop %ax
	ret
	.globl w8_pf16
w8_pf16:
	pushfw
	vmovaps (%rsp), %xmm0
	popfw
	ret
