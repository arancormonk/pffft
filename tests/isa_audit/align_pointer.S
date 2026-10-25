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
// expect: align w8_ptr
	.text
w8_ptr:
	movups (%rdi), %xmm2
	lddqu (%rdi), %xmm3
	movss (%rdi), %xmm4
	vmovups (%rdi), %ymm1
	vmovaps (%rdi), %ymm0
	vzeroupper
	ret
