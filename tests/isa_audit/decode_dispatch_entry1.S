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
// expect: decode base_jx1_in
// expect: isa base_jx1
// expect-image: isa base_jx1
// a branch only re-decoding finds (behind a stray b8 byte) enters the
// dispatch of base_jx1 after its bounds check with index 1: not proven, so
// table entry 1 (vaddps) stays reachable (dispatch first)
	.text
	.globl base_jx1
base_jx1:
	cmp $0, %ecx
	ja .Ldef_base_jx1
	mov %ecx, %ecx
.Lafter_base_jx1:
	lea .Ltab_base_jx1(%rip), %rdx
	movslq (%rdx,%rcx,4), %rax
	add %rdx, %rax
	jmp *%rax
.Lc0_base_jx1:
	ret
.Lc1_base_jx1:
	vaddps %ymm0, %ymm0, %ymm0
	ret
.Ldef_base_jx1:
	ret
	.globl base_jx1_in
base_jx1_in:
	jmp .Lb_base_jx1+1
.Lb_base_jx1:
	.byte 0xb8
	mov $1, %ecx
	jmp .Lafter_base_jx1
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	ret
	RODATA16
	.p2align 2
.Ltab_base_jx1:
	.long .Lc0_base_jx1-.Ltab_base_jx1
