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
// expect: transition w8_abs
// an indirect jump through an absolute address (a 32-bit address relocation
// inside the instruction): not data, and it leaves with a dirty state
// no-image: a 32-bit absolute address does not link into a DLL
	.text
	.globl w8_abs
w8_abs:
	vaddps %ymm0, %ymm0, %ymm0
	jmp *w8_abs_target
	.data
w8_abs_target:
	.quad 0
