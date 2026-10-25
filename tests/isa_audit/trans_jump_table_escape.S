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
// expect: transition w8_jte
// a jump table with an entry outside the function is no local dispatch
	.text
	.globl w8_jte
w8_jte:
	vaddps %ymm0, %ymm0, %ymm0
	cmp $1, %ecx
	ja .Ldefault
	mov %ecx, %ecx
	lea .Ltable(%rip), %rdx
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
	.long w8_jte_other-.Ltable
	.text
w8_jte_other:
	ret
