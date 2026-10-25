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
// expect: decode base_ovl
// expect: isa base_ovl
// expect-image: decode base_ovl
// expect-image: decode base_ovl
// expect-image: isa base_ovl
// a queued path (je .Lavx) and a later decoding from inside it
// (jmp .Lavx+2): both stay decoded, the vaddps is audited
	.text
	.globl base_ovl
base_ovl:
	test %ecx, %ecx
	je .Lavx
	jmp .Lavx+2
.Lavx:
	vaddps %ymm0, %ymm0, %ymm0
	ret
	nop
	push %rax
