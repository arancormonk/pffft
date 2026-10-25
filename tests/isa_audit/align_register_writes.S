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
// expect: align w8_bts
// expect: align w8_cxa
// bts and cmpxchg write registers: an address derived from an aligned stack
// pointer is not known to be aligned after them
	.text
	.globl w8_bts
w8_bts:
	push %rbp
	mov %rsp, %rbp
	and $-32, %rsp
	mov %rsp, %rax
	bts $0, %rax
	vmovaps %ymm0, (%rax)
	vzeroupper
	leave
	ret
	.globl w8_cxa
w8_cxa:
	push %rbp
	mov %rsp, %rbp
	and $-32, %rsp
	mov %rsp, %rax
	lock cmpxchg %rcx, (%rdi)
	vmovaps %ymm0, (%rax)
	vzeroupper
	leave
	ret
