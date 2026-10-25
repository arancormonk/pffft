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
// expect: align w8_lvw
// leavew loads bp from the stack: rbp is no longer known aligned after it
	.text
	.globl w8_lvw
w8_lvw:
	and $-32, %rsp
	mov %rsp, %rbp
	movw $1, (%rbp)
	.byte 0x66, 0xc9        // leavew (clang has no mnemonic for it)
	vmovaps (%rbp), %xmm0
	ret
