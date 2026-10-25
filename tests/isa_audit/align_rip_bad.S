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
// expect: align w8_rip_bad
// expect: align w8_rip_bad
	RODATA16
	.p2align 4
c16:
	.fill 16, 1, 0
c16b:
	.fill 32, 1, 0
	RODATA32
	.p2align 5
c32:
	.fill 64, 1, 0
	.text
w8_rip_bad:
	vmovaps c16b(%rip), %ymm0
	vmovaps c32+4(%rip), %ymm1
	vzeroupper
	ret
