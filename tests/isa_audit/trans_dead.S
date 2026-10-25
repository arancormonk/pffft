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
// a static function nothing refers to never runs; a linked image has no
// such information
// expect-image: transition w8_dead
// expect-image: transition w8_dead
	.text
w8_dead:
	vaddps %ymm0, %ymm0, %ymm0
	addss %xmm1, %xmm1
	ret
