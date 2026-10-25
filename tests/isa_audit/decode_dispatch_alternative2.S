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
// expect: decode base_alt2
// expect: isa base_alt2
// like decode_dispatch_alternative.S, but the alternative decoding starts
// with a nop, so the instruction just before the fictitious bounds check
// does not overlap it; the executed movabs further back does
	.text
	.globl base_alt2
base_alt2:
	mov $1, %ecx
	test %eax, %eax
	je .Lmov2
	jmp .Lmov2+2
.Lmov2:
	// movabs $imm, %rax: the immediate is "90 48 83 f9 00 77 11 90"
	// (nop; cmp $0,%rcx; ja +0x11; nop)
	.byte 0x48, 0xb8, 0x90, 0x48, 0x83, 0xf9, 0x00, 0x77, 0x11, 0x90
	lea .Ltab_alt2(%rip), %rdx
	movslq (%rdx,%rcx,4), %rax
	add %rdx, %rax
	jmp *%rax
.Lc0_alt2:
	ret
.Lc1_alt2:
	vaddps %ymm0, %ymm0, %ymm0
	ret
	RODATA16
	.p2align 2
.Ltab_alt2:
	.long .Lc0_alt2-.Ltab_alt2
	.long .Lc1_alt2-.Ltab_alt2
