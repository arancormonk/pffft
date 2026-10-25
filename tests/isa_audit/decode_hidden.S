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
// expect: decode base_hid
// expect-image: isa base_hid
// (a linked image is decoded again from the branch target: the hidden
// vaddps is then decoded and audited)
// a branch into the middle of an instruction: the bytes it reaches (here
// vaddps hidden in an immediate) were never decoded, so they are a finding
	.text
	.globl base_hid
base_hid:
	jmp .Lmid+2
.Lmid:
	movabs $0x909090c3c058fcc5, %rax
	ret
