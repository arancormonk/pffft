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
// expect: transition w8_jl32
// a 32-bit lea of the table truncates its address: not the table the
// entries are read from, so the dispatch is not proven
	.text
	.globl w8_jl32
w8_jl32:
	vaddps %ymm0, %ymm0, %ymm0
	cmp $1, %ecx
	ja .Ldefault
	mov %ecx, %ecx
	lea .Ltable(%rip), %edx
	movslq (%rdx,%rcx,4), %rax
	add %rdx, %rax
	jmp *%rax
.Lcase0:
	vzeroupper
	ret
.Lcase1:
	vzeroupper
	ret
.Ldefault:
	vzeroupper
	ret
	RODATA16
	.p2align 2
.Ltable:
	.long .Lcase0-.Ltable
	.long .Lcase1-.Ltable
