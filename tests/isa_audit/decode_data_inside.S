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
// expect: decode base_mid
// expect: decode base_mid
// expect-image: none
// data inside a function with code after it (MSVC debug builds put jump
// tables there) gets the linear decoding out of step: here a stray mov
// opcode byte after a jump swallows the start of the code after it. A
// linked image is decoded again from each branch target: no finding; in
// objects, where every decoded instruction is checked, it stays a finding.
	.text
	.globl base_mid
base_mid:
	test %ecx, %ecx
	jne .Lc1
	jmp .Lc0
	.byte 0xb8
.Lc0:
	ret
.Lc1:
	ret
	nop
	nop
	nop
	ret
