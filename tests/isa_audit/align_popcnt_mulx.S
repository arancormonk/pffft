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
// expect: align w8_pc
// expect: isa w8_mx
// expect: align w8_mx
// popcnt and mulx write their destinations (not the stack pointer, not
// rax/rdx): an aligned register they overwrite is no longer known aligned
// (BMI2 is only allowed in MSVC's separate unit: an isa finding for mulx)
	.text
	.globl w8_pc
w8_pc:
	push %rbp
	mov %rsp, %rbp
	and $-32, %rsp
	mov %rsp, %rbx
	popcnt %rax, %rbx
	vmovaps %ymm0, (%rbx)
	vzeroupper
	leave
	ret
	.globl w8_mx
w8_mx:
	push %rbp
	mov %rsp, %rbp
	and $-32, %rsp
	mov %rsp, %rbx
	mulx %rcx, %rax, %rbx
	vmovaps %ymm0, (%rbx)
	vzeroupper
	leave
	ret
