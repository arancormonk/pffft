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
// expect: transition w8_jcx
// cmpxchg loads the accumulator on a mismatch: the checked index can change
// before the table load, so the dispatch is not proven
	.text
	.globl w8_jcx
w8_jcx:
	vaddps %ymm0, %ymm0, %ymm0
	cmp $1, %eax
	ja .Ldefault
	mov %eax, %eax
	lock cmpxchg %rdi, (%rsi)
	lea .Ltable(%rip), %rdx
	movslq (%rdx,%rax,4), %rcx
	add %rdx, %rcx
	jmp *%rcx
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
