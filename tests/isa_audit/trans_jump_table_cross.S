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
// expect: transition w8_jtx
// another function enters the dispatch after its bounds check with an index
// the check would have rejected: the dispatch is not proven
	.text
	.globl w8_jtx
w8_jtx:
	vaddps %ymm0, %ymm0, %ymm0
	cmp $1, %ecx
	ja .Ldefault
	mov %ecx, %ecx
.Lafter_check:
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
	.long .Lcase1-.Ltable
	.text
	.globl w8_enter
w8_enter:
	mov $2, %ecx
	jmp .Lafter_check
