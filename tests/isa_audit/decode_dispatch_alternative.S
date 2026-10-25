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
// expect: decode base_alt
// expect: isa base_alt
// a dispatch whose bounds check exists only in an alternative decoding: a
// branch into the middle of a movabs makes its immediate decode as
// "cmp $0, %rcx; ja", but the movabs is what runs, so rcx = 1 reaches table
// entry 1 (vaddps). The proof must not use the alternative decoding.
	.text
	.globl base_alt
base_alt:
	mov $1, %ecx
	test %eax, %eax
	je .Lmov
	jmp .Lmov+2
.Lmov:
	// movabs $imm, %rax: the immediate is "48 83 f9 00 77 12 90 90"
	// (cmp $0,%rcx; ja +0x12; nop; nop)
	.byte 0x48, 0xb8, 0x48, 0x83, 0xf9, 0x00, 0x77, 0x12, 0x90, 0x90
	lea .Ltab_alt(%rip), %rdx
	movslq (%rdx,%rcx,4), %rax
	add %rdx, %rax
	jmp *%rax
.Lc0_alt:
	ret
.Lc1_alt:
	vaddps %ymm0, %ymm0, %ymm0
	ret
	RODATA16
	.p2align 2
.Ltab_alt:
	.long .Lc0_alt-.Ltab_alt
	.long .Lc1_alt-.Ltab_alt
