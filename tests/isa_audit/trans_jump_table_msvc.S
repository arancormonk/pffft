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
// expect: none
// MSVC's jump tables: lea __ImageBase; the entries are RVAs, in the code
// after the function (COFF: the proven table is data; elsewhere a relative
// table). With an aligned access in the same function.
	.text
	.globl w8_mjt
w8_mjt:
	subq $40, %rsp
	vaddps %ymm0, %ymm0, %ymm0
	vmovaps %xmm0, (%rsp)
	cmpl $1, %ecx
	ja .Ldef
	movslq %ecx, %rax
#if defined(_WIN32)
	leaq __ImageBase(%rip), %rdx
	// movl .Ltab@imgrel(%rdx,%rax,4), %ecx, spelled out: binutils before
	// 2.42 has no @imgrel operands
	.byte 0x8b, 0x8c, 0x82
	.rva .Ltab
#else
	leaq .Ltab(%rip), %rdx
	movslq (%rdx,%rax,4), %rcx
#endif
	addq %rdx, %rcx
	jmpq *%rcx
.Lc0:
	vzeroupper
	addq $40, %rsp
	ret
.Lc1:
	vzeroupper
	addq $40, %rsp
	ret
.Ldef:
	vzeroupper
	addq $40, %rsp
	ret
#if !defined(_WIN32)
	RODATA16
#endif
	.p2align 2
.Ltab:
#if defined(_WIN32)
	.rva .Lc0
	.rva .Lc1
#else
	.long .Lc0-.Ltab
	.long .Lc1-.Ltab
#endif
