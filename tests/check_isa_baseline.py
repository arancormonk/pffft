#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Check the instruction-set contract of a baseline x86-64 build.

The 8-lane AVX2 backend (functions named pffft_w8_* or w8_*) only runs
after the CPU check; everything else must run on any x86-64 CPU. For each
function of the given objects, archives or binaries (disassembled with GNU
objdump -d -r -w; ELF and PE/COFF), three rules are checked:

isa         Outside the 8-lane backend only baseline x86-64 (SSE2)
            instructions. Inside it only the extensions the CPU probe
            checks: the PFFFT_X86_WIDE_ISA_* entries of src/pffft_cpu_x86.h
            (fma is left out with --no-fma). VEX, EVEX and XOP are told
            apart by their encoding, so VEX-encoded BMI (andn, shlx, ...)
            is caught; legacy-encoded extensions (SSE3..SSE4.2, POPCNT,
            LZCNT, TZCNT, MOVBE, ...) by mnemonic.
align       Every instruction that requires an aligned memory operand, in
            the 8-lane backend, and every 32- or 64-byte one anywhere, must
            provably access an aligned address: a stack slot whose offset
            is known modulo 32 (tracked from rsp = 8 mod 16 at entry), or
            RIP-relative data whose section alignment and offset (with the
            relocation applied) suffice. Anything else is reported.
transition  In the 8-lane backend: no legacy-SSE instruction, no call or
            tail jump to other code and no return to other code while the
            upper halves of the ymm registers may be dirty. Interprocedural:
            the state flows through calls between 8-lane functions.

A fourth, decode, makes the others sound: bytes objdump cannot decode, and
branches to an address inside a function that is not the start of an
instruction it decoded (code hidden in another instruction's bytes), are
findings. An indirect jump counts as a local jump table only when the
dispatch is proven (a bounds check, the table's address, the load and the
jump register, in a straight line no branch from any function enters); its
entries are read from the relocations or the image. Both that proof and the
stack tracking of the align rule use one model of the registers each
instruction writes, implicit writes included (written()); an instruction it
does not know invalidates the proof and everything tracked. Conditional
branches out of a function are tail exits on their taken edge.

Code at an address with several names (identical functions folded by the
linker, aliases) is 8-lane code for the isa rule only if all of its names
are, and is checked by the other rules if any is.

--wide-tu also allows the rest of x86-64-v3 in the 8-lane backend, which MSVC
compiles with /arch:AVX2 as its own translation unit.
--skip-empty skips objects without machine code: MSVC /GL ones and LLVM
bitcode (LTO), recognised by their headers; one file must still hold machine
code unless --all-il-ok (the linked image is audited instead).
--map audits a linked MSVC image (exe or DLL, which carry no symbols) by the
functions of its linker map whose Lib:Object matches --map-objects; with
LTCG this is the only way to see the generated code. Only the instructions
reachable from a function's entry are checked (MSVC puts jump tables into
the code). --require names functions that must be among those audited;
--require-unless-ltcg names more, unless an object of --ltcg-objects (the
library's objects) holds no machine code (link-time code generation may
inline them); without link-time code generation only the functions of those
objects are audited (not all the code they call: MSVC may pick the copy of
an inline CRT helper, such as printf, from another object; the object audit
covers the library's own code). --require-wide needs some 8-lane function among them.
--no-transitions skips the third rule, a performance rule, for unoptimized
builds (MSVC /Od compiles scalar code with legacy SSE encodings).

Only meaningful for builds configured for the baseline ISA
(TARGET_C_ARCH=none); a host-tuned build legitimately uses AVX everywhere.
The exemption is by name: code that runs before the CPU check must not use
the 8-lane prefixes.

usage: check_isa_baseline.py [--objdump PATH] [--no-fma] [--no-transitions] [--wide-tu]
                             [--skip-empty [--all-il-ok]] [--map MAP [--map-objects REGEX]]
                             [--require NAME,...] [--require-unless-ltcg NAME,...
                             --ltcg-objects FILE;...] [--require-wide] FILE...
       check_isa_baseline.py --self-test --cc CC [--cc-arg ARG]... [--objdump PATH]
                             [--linker LINK]
       (exit status 1 on findings or when a file holds no functions)
       --self-test with --linker (link.exe or lld-link, COFF fixtures) also
       links each fixture into a DLL and audits it by its map.
"""
import os
import re
import subprocess
import sys
import tempfile

WIDE_PREFIXES = ("pffft_w8_", "w8_")
HERE = os.path.dirname(os.path.abspath(__file__))
CPU_HEADER = os.path.join(HERE, "..", "src", "pffft_cpu_x86.h")


def wide(name):
    return name.startswith(WIDE_PREFIXES)


# ---------------------------------------------------------------------------
# instruction classification

LEGACY_PREFIXES = {0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3}
MNEMONIC_PREFIXES = {"notrack", "bnd", "rep", "repz", "repnz", "repe", "repne", "lock",
                     "data16", "addr32", "cs", "ds", "ss", "es", "fs", "gs", "{vex}", "{vex3}",
                     "{evex}", "rex", "rex.w", "rex.b", "rex.x", "rex.r"}

# legacy (non-VEX) encodings beyond x86-64's SSE2 baseline
LEGACY_EXT = {
    "sse3": {"addsubpd", "addsubps", "haddpd", "haddps", "hsubpd", "hsubps", "lddqu",
             "movddup", "movshdup", "movsldup", "fisttp", "fisttps", "fisttpl", "fisttpll",
             "monitor", "mwait"},
    "ssse3": {"pabsb", "pabsw", "pabsd", "palignr", "phaddw", "phaddd", "phaddsw", "phsubw",
              "phsubd", "phsubsw", "pmaddubsw", "pmulhrsw", "pshufb", "psignb", "psignw",
              "psignd"},
    "sse4.1": {"blendpd", "blendps", "blendvpd", "blendvps", "dppd", "dpps", "extractps",
               "insertps", "movntdqa", "mpsadbw", "packusdw", "pblendvb", "pblendw", "pcmpeqq",
               "pextrb", "pextrd", "pextrq", "phminposuw", "pinsrb", "pinsrd", "pinsrq",
               "pmaxsb", "pmaxsd", "pmaxud", "pmaxuw", "pminsb", "pminsd", "pminud", "pminuw",
               "pmuldq", "pmulld", "ptest", "roundpd", "roundps", "roundsd", "roundss"},
    "sse4.2": {"pcmpestri", "pcmpestrm", "pcmpistri", "pcmpistrm", "pcmpgtq", "crc32",
               "crc32b", "crc32w", "crc32l", "crc32q"},
    "popcnt": {"popcnt", "popcntw", "popcntl", "popcntq"},
    "lzcnt": {"lzcnt", "lzcntw", "lzcntl", "lzcntq"},
    "bmi1": {"tzcnt", "tzcntw", "tzcntl", "tzcntq"},
    "movbe": {"movbe", "movbew", "movbel", "movbeq"},
    "adx": {"adcx", "adox", "adcxl", "adcxq", "adoxl", "adoxq"},
    "aes": {"aesenc", "aesenclast", "aesdec", "aesdeclast", "aesimc", "aeskeygenassist"},
    "pclmul": {"pclmulqdq", "pclmullqlqdq", "pclmulhqlqdq", "pclmullqhqdq", "pclmulhqhqdq"},
    "sha": {"sha1rnds4", "sha1nexte", "sha1msg1", "sha1msg2", "sha256rnds2", "sha256msg1",
            "sha256msg2"},
    "xsave": {"xsave", "xsave64", "xrstor", "xrstor64", "xsetbv"},
    # separate CPUID bits (leaf 0Dh, subleaf 1), which the probe does not check
    "xsaveopt": {"xsaveopt", "xsaveopt64"},
    "xsavec": {"xsavec", "xsavec64"},
    "xsaves": {"xsaves", "xsaves64", "xrstors", "xrstors64"},
    "cx16": {"cmpxchg16b"},
    "prfchw": {"prefetchw"},
    "rdrand": {"rdrand"}, "rdseed": {"rdseed"},
}
LEGACY_EXT_PREFIX = {"pmovsx": "sse4.1", "pmovzx": "sse4.1"}
BMI1_VEX = {"andn", "bextr", "blsi", "blsmsk", "blsr"}
BMI2_VEX = {"bzhi", "mulx", "pdep", "pext", "rorx", "sarx", "shlx", "shrx"}
# VEX-encoded instructions that need a CPUID bit other than AVX/AVX2/FMA
VEX_OTHER = (("vcvtph2ps", "f16c"), ("vcvtps2ph", "f16c"), ("vaes", "vaes"),
             ("vpclmul", "pclmul"), ("vgf2p8", "gfni"), ("vpdpb", "avxvnni"),
             ("vpdpw", "avxvnni"), ("vpmadd52", "avxifma"), ("vcvtne", "avxneconvert"),
             ("vbcstne", "avxneconvert"), ("vsha512", "sha512"), ("vsm3", "sm3"),
             ("vsm4", "sm4"))
FMA3 = re.compile(r"^vf(n?)m(add|sub|addsub|subadd)(132|213|231)")
FMA4 = re.compile(r"^vf(n?)m(add|sub|addsub|subadd)(ps|pd|ss|sd)$")


def encoding(code):
    i = 0
    while i < len(code) and code[i] in LEGACY_PREFIXES:
        i += 1
    if i < len(code) and 0x40 <= code[i] <= 0x4F:
        i += 1
    if i >= len(code):
        return "legacy"
    b = code[i]
    # in 64-bit mode C4/C5 are always VEX and 62 always EVEX
    if b in (0xC4, 0xC5):
        return "vex"
    if b == 0x62:
        return "evex"
    if b == 0x8F and i + 1 < len(code) and ((code[i + 1] >> 3) & 7) != 0:
        return "xop"
    return "legacy"


def extension(insn):
    """the CPUID feature an instruction needs beyond SSE2, or None"""
    enc, m = insn.enc, insn.mnem
    if enc == "evex":
        return "avx512"
    if enc == "xop":
        return "xop"
    if enc == "vex":
        if m in BMI1_VEX:
            return "bmi1"
        if m in BMI2_VEX:
            return "bmi2"
        if FMA3.match(m):
            return "fma"
        if FMA4.match(m):
            return "fma4"
        for prefix, feature in VEX_OTHER:
            if m.startswith(prefix):
                return feature
        if not m.startswith("v"):
            return "vex-other"      # mask registers (AVX-512), AMX, ...
        return "avx"                # AVX and AVX2 are not told apart
    for feature, names in LEGACY_EXT.items():
        if m in names:
            return feature
    for prefix, feature in LEGACY_EXT_PREFIX.items():
        if m.startswith(prefix):
            return feature
    return None


def wide_features(header, fma, tu=False):
    """the extensions the probe checks, from the PFFFT_X86_WIDE_ISA table
    (with tu, also the entries for MSVC's /arch:AVX2 translation unit)"""
    text = open(header).read()
    names = re.findall(r'PFFFT_X86_WIDE_ISA_ENTRY\(\s*"([a-z0-9.]+)"', text)
    if tu:
        names += re.findall(r'PFFFT_X86_WIDE_ISA_TU_ENTRY\(\s*"([a-z0-9.]+)"', text)
    if not names:
        raise SystemExit(f"{header}: no PFFFT_X86_WIDE_ISA_ENTRY found")
    feats = set(names)
    if "avx2" in feats:
        feats.add("avx")
    if not fma:
        feats.discard("fma")
    return feats


# registers
XMM = re.compile(r"%xmm\d+")
YMM = re.compile(r"%[yz]mm\d+")
GPR64 = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11",
         "r12", "r13", "r14", "r15"}
GPR_ALIAS = {}
for _r in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp"):
    GPR_ALIAS["e" + _r] = "r" + _r
    GPR_ALIAS[_r] = "r" + _r
for _n in range(8, 16):
    for _s in ("d", "w", "b"):
        GPR_ALIAS[f"r{_n}{_s}"] = f"r{_n}"
for _r, _b in (("al", "rax"), ("bl", "rbx"), ("cl", "rcx"), ("dl", "rdx"), ("sil", "rsi"),
               ("dil", "rdi"), ("bpl", "rbp"), ("spl", "rsp"), ("ah", "rax"), ("bh", "rbx"),
               ("ch", "rcx"), ("dh", "rdx")):
    GPR_ALIAS[_r] = _b
for _r in GPR64:
    GPR_ALIAS[_r] = _r
MEM = re.compile(r"(?:%[a-z]s:)?(-?0x[0-9a-f]+|-?\d+)?\((%[a-z0-9]+)?(?:,(%[a-z0-9]+)(?:,(\d))?)?\)")

ALIGNED_VEX = {"vmovaps", "vmovapd", "vmovdqa", "vmovdqa32", "vmovdqa64", "vmovntps",
               "vmovntpd", "vmovntdq", "vmovntdqa"}
UNALIGNED_LEGACY = {"movups", "movupd", "movdqu", "lddqu", "movss", "movsd", "movd", "movq",
                    "movlps", "movhps", "movlpd", "movhpd", "movddup", "cvtps2pd",
                    "cvtdq2pd", "comiss", "comisd", "ucomiss", "ucomisd", "insertps",
                    "extractps", "maskmovdqu", "movnti", "ldmxcsr", "stmxcsr", "pinsrw",
                    "pinsrb", "pinsrd", "pinsrq", "pextrw", "pextrb", "pextrd", "pextrq",
                    "cvtpi2ps", "cvtpi2pd", "cvtps2pi", "cvttps2pi", "cvtpd2pi", "cvttpd2pi"}


def alignment_required(insn):
    """bytes of alignment the memory operand must have, or 0"""
    if insn.mem is None:
        return 0
    m = insn.mnem
    if insn.enc in ("vex", "evex"):
        if m in ALIGNED_VEX:
            if "%zmm" in insn.ops:
                return 64
            return 32 if "%ymm" in insn.ops else 16
        return 0
    if not XMM.search(insn.ops):
        return 0
    if m in UNALIGNED_LEGACY or m.startswith(("pmovsx", "pmovzx", "cvtsi2s", "cvtss2", "cvtsd2",
                                              "cvttss2", "cvttsd2", "prefetch")):
        return 0
    if not m.startswith("p") and (m.endswith("ss") or m.endswith("sd")):
        return 0          # scalar single/double: 32- or 64-bit operand
    return 16             # packed SSE with a 128-bit memory operand


# ---------------------------------------------------------------------------
# objdump parsing

class Insn:
    __slots__ = ("addr", "code", "text", "mnem", "ops", "enc", "relocs", "target",
                 "target_name", "mem", "size", "bad")

    def __init__(self, addr, code, text):
        self.addr, self.code, self.text = addr, code, text
        self.size = len(code)
        self.relocs = []
        body = text.split("#", 1)[0]
        tokens = body.split()
        while tokens and tokens[0].lower() in MNEMONIC_PREFIXES:
            tokens.pop(0)
        self.mnem = tokens[0].lower() if tokens else ""
        self.ops = " ".join(tokens[1:])
        self.enc = encoding(code)
        self.target = self.target_name = None
        t = re.match(r"^([0-9a-f]+) <([^>+]+)(\+0x[0-9a-f]+)?>", self.ops)
        if t:
            self.target = int(t.group(1), 16)
            self.target_name = t.group(2) if t.group(3) is None else None
        elif re.fullmatch(r"(0x)?[0-9a-f]+", self.ops) and (self.mnem.startswith(("j", "loop", "call"))):
            self.target = int(self.ops, 16)
        mm = MEM.search(self.ops)
        self.mem = mm.groups() if mm and "(" in self.ops else None
        # bytes objdump could not decode (or only as data)
        self.bad = self.mnem in ("(bad)", ".byte", ".word", ".long", ".quad") or "(bad)" in body


class Function:
    def __init__(self, unit, section, name, addr, aliases=()):
        self.unit, self.section, self.addr = unit, section, addr
        names = sorted(set(aliases) | {name})
        # the name findings are reported under: a baseline one if any
        self.name = next((n for n in names if not wide(n)), names[0]) if len(names) > 1 else name
        self.aliases = names
        self.section_number = None       # COFF objects: from the symbol table
        self.end = None                  # the end of its bytes (set by the parsers)
        self.insns = []

    @property
    def wide_isa(self):
        """may use the 8-lane backend's extensions: only if no baseline code
        can reach it under another name"""
        return all(wide(n) for n in self.aliases)

    @property
    def wide_flow(self):
        """8-lane code for the alignment and transition rules"""
        return any(wide(n) for n in self.aliases)

    def span_end(self):
        if self.end is not None:
            return self.end
        return self.insns[-1].addr + self.insns[-1].size if self.insns else self.addr


INSN = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*\t?(.*)$")
RELOC_INLINE = re.compile(r"\t([0-9a-f]+): ((?:R_X86_64|IMAGE_REL_AMD64)_[A-Z0-9_]+)\t(\S+)")
RELOC_LINE = re.compile(r"^\s+([0-9a-f]+): ((?:R_X86_64|IMAGE_REL_AMD64)_[A-Z0-9_]+)\t(\S+)")


def run(objdump, *args):
    return subprocess.run([objdump, *args], check=True, capture_output=True, text=True).stdout


def split_symbol(s):
    """'sym+0x10' / 'sym-0x4' -> (sym, addend)"""
    m = re.match(r"^(.+?)([+-]0x[0-9a-f]+)?$", s)
    return m.group(1), int(m.group(2), 16) if m.group(2) else 0


def parse(objdump, path):
    """[Function], per-unit symbol and section tables"""
    funcs, unit, section, cur = [], path, None, None
    for line in run(objdump, "-d", "-r", "-w", path).splitlines():
        m = re.match(r"^(\S.*):\s+file format (\S+)$", line)
        if m:
            unit = m.group(1); cur = None; continue
        m = re.match(r"^Disassembly of section (\S+):$", line)
        if m:
            section = m.group(1); cur = None; continue
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:$", line)
        if m:
            cur = Function(unit, section, m.group(2), int(m.group(1), 16))
            funcs.append(cur); continue
        if cur is None:
            continue
        r = RELOC_LINE.match(line)
        if r and cur.insns:
            cur.insns[-1].relocs.append((int(r.group(1), 16), r.group(2), *split_symbol(r.group(3))))
            continue
        m = INSN.match(line)
        if not m:
            continue
        text = m.group(3)
        relocs = [(int(a, 16), t, *split_symbol(s)) for a, t, s in RELOC_INLINE.findall(text)]
        text = RELOC_INLINE.sub("", text)
        code = [int(b, 16) for b in m.group(2).split()]
        if not text.strip() and cur.insns:
            # continuation of the previous instruction's bytes
            prev = cur.insns[-1]
            prev.code += code; prev.size = len(prev.code)
            continue
        insn = Insn(int(m.group(1), 16), code, text)
        insn.relocs = relocs
        cur.insns.append(insn)
    return funcs


def msvc_map(path, objects):
    """[(address, [names], selected)] of the function starts of an MSVC or
    lld-link linker map, sorted, and the image's preferred load address.
    Function starts: the symbols flagged 'f'; for objects without such
    flags (clang-cl's, assembly) every symbol in a code section. All names
    at an address are kept (identical COMDAT folding). selected: the
    Lib:Object column of one of them matches the regex objects."""
    code_sections, entries, base = set(), [], 0
    for line in open(path, errors="replace"):
        m = re.match(r"^\s*Preferred load address is ([0-9a-f]+)", line)
        if m:
            base = int(m.group(1), 16)
        m = re.match(r"^\s*([0-9a-f]{4}):[0-9a-f]{8}\s+[0-9a-f]{8}H\s+\S+\s+CODE\s*$", line)
        if m:
            code_sections.add(int(m.group(1), 16)); continue
        m = re.match(r"^\s*([0-9a-f]{4}):[0-9a-f]{8}\s+(\S+)\s+([0-9a-f]{8,16})\s+(f\s+)?(?:i\s+)?(\S+)\s*$", line)
        if m:
            entries.append((int(m.group(1), 16), m.group(2), int(m.group(3), 16), bool(m.group(4)),
                            m.group(5)))
    flagged = {obj for _, _, _, f, obj in entries if f}
    funcs = {}
    for sec, name, a, f, obj in entries:
        if f or (sec in code_sections and obj not in flagged):
            names, sel = funcs.get(a, ([], False))
            hit = objects(obj) if callable(objects) else re.search(objects, obj) is not None
            funcs[a] = (names + [name], sel or hit)
    return sorted((a, names, sel) for a, (names, sel) in funcs.items()), base


def decode_range(objdump, path, start, stop):
    """[Insn] objdump decodes from start (up to stop) in a linked image"""
    out = []
    text = run(objdump, "-d", "-w", f"--start-address={start:#x}", f"--stop-address={stop:#x}", path)
    for line in text.splitlines():
        m = INSN.match(line)
        if not m:
            continue
        code = [int(b, 16) for b in m.group(2).split()]
        if not m.group(3).strip() and out:
            out[-1].code += code; out[-1].size = len(out[-1].code)
            continue
        out.append(Insn(int(m.group(1), 16), code, m.group(3)))
    return out


def decode_mapped(objdump, path, mapped):
    """the selected functions of a symbol-less image, each decoded from its
    own start (data between functions cannot shift the decoding) up to the
    next function of the map"""
    out, unit = [], path
    for k, (addr, names, sel) in enumerate(mapped):
        if not sel:
            continue
        end = mapped[k + 1][0] if k + 1 < len(mapped) else addr + 0x100000
        f = Function(unit, ".text", names[0], addr, names)
        f.end = end
        f.insns = decode_range(objdump, path, addr, end)
        if f.insns:
            # the last function's end: what objdump decoded
            if k + 1 == len(mapped):
                f.end = f.insns[-1].addr + f.insns[-1].size
            out.append(f)
    return out


def image_bytes(objdump, path, prefixes):
    """{address: byte} of the sections of a linked image whose names start
    with one of prefixes"""
    out, cur = {}, False
    for line in run(objdump, "-s", path).splitlines():
        m = re.match(r"^Contents of section (\S+):$", line)
        if m:
            cur = m.group(1).startswith(prefixes); continue
        m = re.match(r"^ ([0-9a-f]+) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if cur and m:
            base, data = int(m.group(1), 16), bytes.fromhex(m.group(2).replace(" ", ""))
            for k, b in enumerate(data):
                out[base + k] = b
    return out


STRING_OPS = {"movsb", "movsw", "movsl", "movsq", "stos", "stosb", "stosw", "stosl", "stosq",
              "lods", "lodsb", "lodsw", "lodsl", "lodsq", "scas", "scasb", "scasw", "scasl",
              "scasq", "cmpsb", "cmpsw", "cmpsl", "cmpsq", "insb", "insw", "insl", "outsb",
              "outsw", "outsl"}
# instructions whose general-purpose register writes are all implicit
IMPLICIT_WRITES = {
    "cpuid": {"rax", "rbx", "rcx", "rdx"}, "rdtsc": {"rax", "rdx"}, "rdtscp": {"rax", "rcx", "rdx"},
    "xgetbv": {"rax", "rdx"}, "rdpmc": {"rax", "rdx"}, "syscall": {"rax", "rcx", "r11"},
    "cqto": {"rdx"}, "cltd": {"rdx"}, "cwtd": {"rdx"}, "cqo": {"rdx"}, "cdq": {"rdx"}, "cwd": {"rdx"},
    "cltq": {"rax"}, "cdqe": {"rax"}, "cwtl": {"rax"}, "cwde": {"rax"}, "cbtw": {"rax"}, "cbw": {"rax"},
    "lahf": {"rax"}, "leave": {"rsp", "rbp"}, "leaveq": {"rsp", "rbp"}, "leavew": {"rsp", "rbp"},
    "pcmpestri": {"rcx"}, "pcmpistri": {"rcx"}, "vpcmpestri": {"rcx"}, "vpcmpistri": {"rcx"},
    "cmpxchg8b": {"rax", "rdx"}, "cmpxchg16b": {"rax", "rdx"},
}
NO_WRITES = {"sahf", "pause", "lfence", "mfence", "sfence", "int3", "ud2", "hlt", "endbr64",
             "clc", "stc", "cld", "std", "cmc", "vzeroupper", "vzeroall", "emms", "ldmxcsr",
             "stmxcsr", "vldmxcsr", "vstmxcsr", "fwait", "wait"}
# exact families (with their operand-size suffixes) whose names are prefixes
# of others: mulx is no mul, popcnt no pop, pushf/popf are
MULX = re.compile(r"^mulx[lq]?$")
MUL_DIV = re.compile(r"^(mul|div|idiv)[bwlq]?$")
IMUL = re.compile(r"^imul[bwlq]?$")
PUSH = re.compile(r"^push(f)?[wlq]?$")
POP = re.compile(r"^pop(f)?[wlq]?$")
# families whose only general-purpose register write is the last operand
WRITES_LAST = re.compile(r"^(mov|lea|add|sub|adc|sbb|and|or|xor|sh[lr]|sa[lr]|ro[lr]|rc[lr]|inc|dec"
                         r"|neg|not|cmov|set|bswap|bextr|bls|bzhi|pdep|pext|sarx|shlx|shrx|lzcnt"
                         r"|tzcnt|popcnt|crc32|bt[crs])")


def written(insn):
    """the general-purpose registers (64-bit names) an instruction may write,
    explicitly or implicitly, or None when that is not known: only the
    families listed here are understood, everything else counts as unknown"""
    m, ops = insn.mnem, split_operands(insn.ops)
    regs = [reg(o) for o in ops if o.startswith("%") and reg(o)]
    if m in IMPLICIT_WRITES:
        return set(IMPLICIT_WRITES[m])
    if m in NO_WRITES or m.startswith(("nop", "prefetch")):
        return set()
    if m in STRING_OPS:
        return {"rdi", "rsi", "rcx", "rax"}
    if m.startswith("loop"):
        return {"rcx"}
    if m.startswith("j") or m in RET:
        return set()                          # control leaves; no register written
    if m.startswith("call"):
        return set(CALLER_SAVED)              # what the callee may change
    if m.startswith(("cmpxchg", "xadd", "xchg")):
        # cmpxchg loads the accumulator on a mismatch; xadd and xchg write both
        return set(regs) | ({"rax"} if m.startswith("cmpxchg") else set())
    if m.startswith(("cmp", "test")) or m in ("bt", "btl", "btq", "btw"):
        return set()
    if MULX.match(m):
        return {r for r in (reg(ops[-1]), reg(ops[-2])) if r}
    if MUL_DIV.match(m) or (IMUL.match(m) and len(ops) == 1):
        return {"rax", "rdx"}
    if PUSH.match(m):
        return {"rsp"}
    if POP.match(m):
        return {"rsp"} | ({reg(ops[0])} if ops and reg(ops[0]) else set())
    if m.startswith(("fnstsw", "fstsw")):
        return {"rax"} if ops and reg(ops[0]) == "rax" else set()
    if m.startswith("f") and not regs:
        return set()                          # x87 arithmetic: no general-purpose register
    if not ops:
        return None
    if WRITES_LAST.match(m) or (m.startswith("imul") and len(ops) >= 2) \
            or m.startswith("v") or XMM.search(insn.ops) or YMM.search(insn.ops):
        r = reg(ops[-1]) if ops[-1].startswith("%") else None
        return {r} if r else set()
    return None


def reg_bits(name):
    """32 or 64 for a general-purpose register operand, else 0"""
    n = name.lstrip("%")
    if re.fullmatch(r"e[a-z]{2}|r\d+d", n):
        return 32
    if re.fullmatch(r"r[a-z]{2}|r\d+", n):
        return 64
    return 0


def jump_table_dispatch(f, i, incoming=None):
    """(lea, load, count, kind) of a proven table dispatch ending in the
    register-indirect jump f.insns[i], or None:

        cmp $N, idx; ja/jae default          (adjacent: the bounds check)
        movslq %eJ, %rI / mov %eJ, %eI       (index extensions, optional)
        lea T(%rip), %rB                     (the table, or MSVC's __ImageBase;
                                              before or after the check)
        movslq (%rB,%rI,4), %rX              kind 'rel': entries target - table
          or mov D(%rB,%rI,4), %eX           kind 'rva': entries are RVAs, D the table's
        add %rB, %rX
        jmp *%rX

    as byte-contiguous instructions no other decoding overlaps, with no
    other write to a register involved, no other branch, and no direct
    branch from anywhere (incoming: the targets of all branches that
    can reach f's code, also from other functions) into the sequence after
    its first instruction. A 32-bit bounds check of a 64-bit index needs an extension
    after it or a 32-bit write of the index (which clears the upper half)
    before it, in the same straight line."""
    jmp = f.insns[i]
    t = jmp.ops.lstrip("*").strip()
    if not (is_jump(jmp) and indirect(jmp) and reg_bits(t) == 64):
        return None
    x, b, idx = reg(t), None, None
    add = load = lea = count = kind = None
    first = None
    extended = False
    q = i - 1
    while q >= 0 and i - q <= 24:
        c = f.insns[q]
        m, ops = c.mnem, split_operands(c.ops)
        if count is None and m in ("ja", "jae") and load is not None and idx is not None:
            cmp = f.insns[q - 1] if q > 0 else None
            cops = split_operands(cmp.ops) if cmp is not None else []
            if cmp is None or not cmp.mnem.startswith("cmp") or len(cops) != 2 \
                    or not cops[0].startswith("$") or reg(cops[1]) != idx:
                return None
            bits = reg_bits(cops[1])
            if bits == 0:
                return None
            n = imm(cops[0][1:])
            count = n + 1 if m == "ja" else n
            if not 0 < count <= 4096:
                return None
            first = q - 1
            if bits == 32 and not extended:
                # the index's last write before the check must be a 32-bit
                # one, which clears the upper half
                r = first - 1
                while r >= 0 and first - r <= 24:
                    c2 = f.insns[r]
                    w2 = written(c2)
                    if is_jump(c2) or c2.mnem.startswith("call") or c2.mnem in RET or w2 is None \
                            or (b is not None and b in w2):
                        return None
                    if idx in w2:
                        ops2 = split_operands(c2.ops)
                        if w2 != {idx} or reg_bits(ops2[-1]) != 32:
                            return None
                        break
                    r -= 1
                else:
                    return None
                first = r
            idx, q = None, first - 1
            if lea is not None:
                break
            continue
        if is_jump(c) or m.startswith("call") or m in RET:
            return None
        w = written(c)
        if w is None:
            return None
        if not w & {r for r in (x, b, idx) if r}:
            q -= 1
            continue
        if add is None:
            if m in ("add", "addq") and len(ops) == 2 and reg(ops[1]) == x and reg_bits(ops[0]) == 64 \
                    and reg_bits(ops[1]) == 64 and reg(ops[0]) != x:
                add, b = c, reg(ops[0])
            else:
                return None
        elif load is None:
            if w != {x} or b in w or not (c.mem and c.mem[1] and c.mem[2] and c.mem[3] == "4") \
                    or reg(c.mem[1]) != b or reg(c.mem[2]) in (b, None):
                return None
            disp = imm(c.mem[0]) if c.mem[0] else 0
            if m in ("movslq", "movsxd") and reg_bits(ops[-1]) == 64 and disp == 0:
                kind = "rel"
            elif m in ("mov", "movl") and reg_bits(ops[-1]) == 32:
                kind = "rva"
            else:
                return None
            load, idx, x = c, reg(c.mem[2]), None
        elif b is not None and b in w:
            if lea is None and w == {b} and m in ("lea", "leaq") and c.mem and c.mem[1] == "%rip" \
                    and not c.mem[2] and reg_bits(ops[-1]) == 64:
                lea, b = c, None
                if count is not None:
                    first = min(first, q)
                    break
            else:
                return None
        elif idx is not None and w == {idx} and count is None:
            # an extension of the index, from itself or another register
            if m in ("cltq", "cdqe"):
                extended = True
            elif len(ops) == 2 and reg_bits(ops[0]) == 32 and (
                    (m in ("movslq", "movsxd") and reg_bits(ops[1]) == 64)
                    or (m in ("mov", "movl") and reg_bits(ops[1]) == 32)):
                idx, extended = reg(ops[0]), True
            else:
                return None
        else:
            return None
        q -= 1
    if count is None or lea is None or first is None:
        return None
    # one decoding: the sequence is byte-contiguous and no other decoded
    # instruction overlaps any of its bytes (a linked image keeps alternative
    # decodings side by side)
    if any(f.insns[r].addr + f.insns[r].size != f.insns[r + 1].addr for r in range(first, i)):
        return None
    lo, hi = f.insns[first].addr, jmp.addr + jmp.size
    if any(insn.addr < hi and insn.addr + insn.size > lo
           for r, insn in enumerate(f.insns) if not first <= r <= i):
        return None
    # nothing may enter the sequence after its first instruction
    targets = incoming if incoming is not None else \
        {insn.target for insn in f.insns if insn.target is not None and branch_reloc(insn) is None}
    if any(f.insns[r].addr in targets for r in range(first + 1, i + 1)):
        return None
    return lea, load, count, kind


class Context:
    """what jump tables are resolved from: kind 'elf-object' (syms, drel),
    'coff-object' (coff), 'image' (data: {address: byte}; image_base)"""
    def __init__(self, kind, syms=None, drel=None, coff=None, data=None, image_base=0):
        self.kind, self.syms, self.drel, self.coff = kind, syms or {}, drel or {}, coff
        self.data, self.image_base = data or {}, image_base
        self.incoming = None          # {code_key(f): set of branch targets}

    def incoming_for(self, f):
        return None if self.incoming is None else self.incoming.get(code_key(f, self.kind), set())


def code_key(f, kind):
    """the address space of f's code: an image is one, an object has one per section"""
    if kind == "image":
        return (f.unit,)
    return (f.unit, f.section_number if f.section_number is not None else f.section)


def incoming_targets(funcs, kind, syms=None):
    """{code_key: addresses some direct branch or call goes to}, from every
    function: resolved displacements, and in objects relocated ones whose
    symbol and addend are known (ELF PC32/PLT32, COFF REL32)"""
    out = {}
    for f in funcs:
        usyms = (syms or {}).get(f.unit, {})
        for insn in f.insns:
            if not (is_jump(insn) or insn.mnem.startswith("call")) or insn.target is None:
                continue
            nxt = insn.addr + insn.size
            for off, rtype, sym, addend in insn.relocs:
                if not insn.addr < off < nxt:
                    continue
                if sym in usyms:
                    sec, value = usyms[sym]
                    if rtype.startswith("IMAGE_REL_AMD64_REL32"):
                        k = int(rtype[-1]) if rtype[-2] == "_" else 0
                        t = value + (insn.target - nxt) + nxt - off - 4 - k
                    elif rtype in ("R_X86_64_PC32", "R_X86_64_PLT32"):
                        t = value + addend + nxt - off
                    else:
                        break
                    out.setdefault((f.unit, sec), set()).add(t)
                break
            else:
                out.setdefault(code_key(f, kind), set()).add(insn.target)
    return out


def read32(data, a, signed=False):
    if not all(a + k in data for k in range(4)):
        return None
    return int.from_bytes(bytes(data[a + k] for k in range(4)), "little", signed=signed)


def table_targets(f, i, ctx):
    """the addresses (offsets in f's section, for objects) of a proven jump
    table dispatch at f.insns[i], or None"""
    d = jump_table_dispatch(f, i, ctx.incoming_for(f))
    if d is None:
        return None
    lea, load, count, kind = d
    disp = imm(load.mem[0]) if load.mem[0] else 0
    if ctx.kind == "image":
        lea_to = lea.addr + lea.size + (imm(lea.mem[0]) if lea.mem[0] else 0)
        out = []
        for k in range(count):
            if kind == "rva":
                if not ctx.image_base or lea_to != ctx.image_base:
                    return None
                v = read32(ctx.data, ctx.image_base + disp + 4 * k)
                if v is None:
                    return None
                out.append(ctx.image_base + v)
            else:
                v = read32(ctx.data, lea_to + 4 * k, signed=True)
                if v is None:
                    return None
                out.append(lea_to + v)
        return out
    if ctx.kind == "coff-object":
        return coff_table(f, lea, load, count, kind, ctx.coff)
    if kind != "rel" or not lea.relocs:
        return None
    # ELF object: entries R_X86_64_PC32 against f's section, addend target + 4k
    off, rtype, sym, addend = lea.relocs[0]
    if rtype != "R_X86_64_PC32" or sym not in ctx.syms:
        return None
    sec, value = ctx.syms[sym]
    table = value + addend + (lea.addr + lea.size - off)
    entries = ctx.drel.get(sec, {})
    out = []
    for k in range(count):
        e = entries.get(table + 4 * k)
        if e is None or e[0] != "R_X86_64_PC32" or e[1] != f.section:
            return None
        out.append(e[2] - 4 * k)
    return out


def coff_table(f, lea, load, count, kind, coff, with_range=False):
    """jump table targets (offsets in f's section) of a COFF object: clang's
    'rel' tables (entries REL32 against f's section, relative to the table)
    and MSVC's 'rva' tables (lea __ImageBase; the load's displacement an
    ADDR32NB field naming the table; entries ADDR32NB). with_range: also
    (section, start, end) of the table. None when that cannot be shown."""
    import struct
    if coff is None or f.section_number is None:
        return None
    code, crel = coff.section(f.section_number)

    def field(insn, types):
        o = next((o for o in crel if insn.addr <= o < insn.addr + insn.size), None)
        if o is None or crel[o][0] not in types or o + 4 > insn.addr + insn.size:
            return None
        return o

    fl = field(lea, (4,))                                   # IMAGE_REL_AMD64_REL32
    if fl is None:
        return None
    lsym = crel[fl][1]
    if kind == "rva":
        if coff.symbol_info[lsym][0] != "__ImageBase" or struct.unpack_from("<i", code, fl)[0] != 0:
            return None
        fd = field(load, (3,))                              # IMAGE_REL_AMD64_ADDR32NB
        if fd is None:
            return None
        tsec, tval = coff.symbols[crel[fd][1]]
        table = tval + struct.unpack_from("<i", code, fd)[0]
        entry_type = 3
    else:
        tsec, tval = coff.symbols[lsym]
        table = tval + struct.unpack_from("<i", code, fl)[0] + (lea.addr + lea.size) - (fl + 4)
        entry_type = 4
    if tsec <= 0:
        return None
    tdata, trel = coff.section(tsec)
    targets = []
    for k in range(count):
        e = table + 4 * k
        if e not in trel or trel[e][0] != entry_type or e + 4 > len(tdata):
            return None
        esec, evalue = coff.symbols[trel[e][1]]
        if esec != f.section_number:
            return None
        v = struct.unpack_from("<i", tdata, e)[0]
        # rel: S + field - (P + 4) = target - table; rva: S + field
        targets.append(evalue + v - 4 - 4 * k if kind == "rel" else evalue + v)
    if with_range:
        return targets, (tsec, table, table + 4 * count)
    return targets


def indirect_exit(insn):
    """an indirect jump through a RIP-relative pointer (the import table, a
    GOT entry): it leaves the function"""
    return insn.ops.startswith("*") and "(%rip)" in insn.ops


def prune_unreachable(funcs, ctx, redecode=None):
    """the instructions of a symbol-less image reachable from a function
    entry (MSVC's jump tables are data in the code), across functions, and
    decode findings. A branch into the span of a function that does not land
    on a decoded instruction (code after data inside the function, which
    shifted the linear decoding: MSVC debug builds put jump tables there) is
    decoded again from its target with redecode(start, stop), until that
    decoding meets an earlier one. Decodings are only ever added, never
    replaced, and reachability is computed again, with the jump-table proofs
    seeing the branches of everything decoded so far, until nothing new is
    decoded. Reachable instructions that overlap, and reachable bytes that
    are no instruction, are findings. An indirect jump that is not a proven
    table keeps all of its function."""
    import bisect
    decoded = {id(f): {i.addr: i for i in f.insns} for f in funcs}
    spans = sorted((f.addr, f.span_end(), k) for k, f in enumerate(funcs))
    views = {}

    def func_at(a):
        for lo, hi, k in spans:
            if lo <= a < hi:
                return funcs[k]
        return None

    def refresh(f):
        f.insns = sorted(decoded[id(f)].values(), key=lambda i: i.addr)
        views[id(f)] = [i.addr for i in f.insns]

    def decode_at(f, a):
        """decode f from a until meeting an earlier decoding; True if a now starts an instruction"""
        if redecode is None:
            return False
        got = False
        for n in redecode(a, f.span_end()):
            if n.addr != a and n.addr in decoded[id(f)]:
                break                                   # back in step
            decoded[id(f)].setdefault(n.addr, n)
            got = got or n.addr == a
        refresh(f)
        return got

    for f in funcs:
        refresh(f)
    while True:
        # every branch of everything decoded so far can enter a dispatch
        ctx.incoming = incoming_targets(funcs, "image")
        grew = False
        reach = {id(f): set() for f in funcs}
        problems = {}
        work = [(f, f.addr) for f in funcs if f.insns]

        def go(f, a, insn):
            g = func_at(a)
            if g is not None:
                work.append((g, a))

        while work:
            f, a = work.pop()
            if a in reach[id(f)]:
                continue
            if a not in decoded[id(f)]:
                if not decode_at(f, a):
                    problems[(id(f), a)] = ("decode", f, f.insns[max(0, bisect.bisect_left(views[id(f)], a) - 1)],
                                            f"branch to {a:#x}, not the start of a decoded instruction")
                    continue
                grew = True
            reach[id(f)].add(a)
            insn = decoded[id(f)][a]
            k = bisect.bisect_left(views[id(f)], a)
            nxt = insn.addr + insn.size
            m = insn.mnem
            if insn.bad:
                problems[(id(f), a)] = ("decode", f, insn, "bytes that are no instruction")
                continue
            if m in RET or m in ("ud2", "hlt"):
                continue
            if is_jump(insn) and indirect(insn):
                if indirect_exit(insn):
                    continue
                t = table_targets(f, k, ctx) if insn.ops.startswith("*%") else None
                if t is None:
                    work.extend((f, i.addr) for i in f.insns)
                else:
                    for b in t:
                        go(f, b, insn)
                continue
            if (is_jump(insn) or m.startswith("call")) and insn.target is not None:
                go(f, insn.target, insn)
                if m in UNCOND:
                    continue
            go(f, nxt, insn)
        if not grew:
            break
    for f in funcs:
        kept = sorted((decoded[id(f)][a] for a in reach[id(f)]), key=lambda i: i.addr)
        for p, q in zip(kept, kept[1:]):
            if p.addr + p.size > q.addr:
                problems[(id(f), q.addr)] = ("decode", f, q,
                                             f"reachable instructions overlap ({p.addr:#x} and {q.addr:#x})")
        f.insns = kept
    return [f for f in funcs if f.insns], list(problems.values())


def decode_findings(funcs):
    """for objects and images with symbols (all decoded instructions are
    checked): bytes that are no instruction, and direct branches into a
    function's span that do not land on a decoded instruction"""
    starts, spans = {}, {}
    for f in funcs:
        k = (f.unit, f.section, f.section_number)
        starts.setdefault(k, set()).update(insn.addr for insn in f.insns)
        spans.setdefault(k, []).append((f.addr, f.span_end()))
    out = []
    for f in funcs:
        k = (f.unit, f.section, f.section_number)
        for insn in f.insns:
            if insn.bad:
                out.append(("decode", f, insn, "bytes that are no instruction"))
            elif (is_jump(insn) or insn.mnem.startswith("call")) and insn.target is not None \
                    and branch_reloc(insn) is None and insn.target not in starts[k] \
                    and any(lo <= insn.target < hi for lo, hi in spans[k]):
                out.append(("decode", f, insn,
                            f"branch to {insn.target:#x}, not the start of a decoded instruction"))
    return out


IL_MAGIC = bytes.fromhex("38feb30ca5d9ab4dac9bd6b6222653c2")       # MSVC /GL (LLVM's ClGlObjMagic)
BIGOBJ_MAGIC = bytes.fromhex("c7a1bad1eebaa94baf20faf66aa4dcb8")


def object_kind(path):
    """'il' (MSVC /GL), 'bitcode' (LLVM, also wrapped), 'bigobj' or 'other'
    from a file's header"""
    with open(path, "rb") as fh:
        head = fh.read(32)
    if head[:4] == b"\x00\x00\xff\xff" and len(head) >= 28:
        if head[12:28] == IL_MAGIC:
            return "il"
        if head[12:28] == BIGOBJ_MAGIC:
            return "bigobj"
    if head[:4] in (b"BC\xc0\xde", b"\xde\xc0\x17\x0b"):
        return "bitcode"
    return "other"


def tables(objdump, path):
    """{unit: {symbol: (section, value)}}, {unit: {section: alignment}},
    {unit: set of local (static) symbols}, {(unit, section, value): [names]}
    of the ELF function symbols"""
    syms, secs, local, fsyms, unit = {}, {}, {}, {}, path
    for line in run(objdump, "-t", "-w", path).splitlines():
        m = re.match(r"^(\S.*):\s+file format", line)
        if m:
            unit = m.group(1); continue
        p = line.split()
        # ELF: value flags... section size name
        if len(p) >= 4 and re.fullmatch(r"[0-9a-f]{8,16}", p[0]):
            sec = next((x for x in p[1:-1] if x.startswith((".", "*")) and x != "*ABS*"), None)
            if sec:
                syms.setdefault(unit, {})[p[-1]] = (sec, int(p[0], 16))
                if "F" in p[1:-3]:
                    fsyms.setdefault((unit, sec, int(p[0], 16)), []).append(p[-1])
            if p[1] == "l":
                local.setdefault(unit, set()).add(p[-1])
            continue
        # COFF: [ n](sec  k)(fl ..)(ty ..)(scl ..) (nx ..) 0xvalue name
        m = re.match(r"^\[\s*\d+\]\(sec\s+(-?\d+)\).*\(scl\s+(\d+)\).*\s0x([0-9a-f]+)\s+(\S+)$", line)
        if m:
            syms.setdefault(unit, {})[m.group(4)] = (int(m.group(1)), int(m.group(3), 16))
            if m.group(2) in ("3", "6"):            # IMAGE_SYM_CLASS_STATIC, _LABEL
                local.setdefault(unit, set()).add(m.group(4))
    unit = path
    for line in run(objdump, "-h", "-w", path).splitlines():
        m = re.match(r"^(\S.*):\s+file format", line)
        if m:
            unit = m.group(1); continue
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+[0-9a-f]+\s+[0-9a-f]+\s+2\*\*(\d+)", line)
        if m:
            d = secs.setdefault(unit, {})
            d[m.group(2)] = (1 << int(m.group(4)))
            d[int(m.group(1)) + 1] = (1 << int(m.group(4)))     # COFF section numbers
            d[("name", int(m.group(1)) + 1)] = m.group(2)
            d[("vma", m.group(2))] = int(m.group(3), 16)
    return syms, secs, local, fsyms


def data_relocations(objdump, path):
    """{unit: {section: {offset: (type, symbol, addend)}}} of all sections"""
    out, unit, section = {}, path, ""
    for line in run(objdump, "-r", "-w", path).splitlines():
        m = re.match(r"^(\S.*):\s+file format", line)
        if m:
            unit = m.group(1); continue
        m = re.match(r"^RELOCATION RECORDS FOR \[(.+)\]:", line)
        if m:
            section = m.group(1); continue
        m = re.match(r"^([0-9a-f]+)\s+(\S+)\s+(\S+)$", line)
        if m:
            sym, addend = split_symbol(m.group(3))
            out.setdefault(unit, {}).setdefault(section, {})[int(m.group(1), 16)] = (m.group(2), sym, addend)
    return out


class Coff:
    """sections, relocations and symbols of a plain COFF object (x86-64),
    read directly: MSVC names all its COMDAT code sections .text$mn, and
    GNU objdump then puts one section's symbols into the others too"""
    def __init__(self, path):
        import struct
        data = open(path, "rb").read()
        machine, nsec, _, symptr, nsym, opt, _ = struct.unpack_from("<HHIIIHH", data, 0)
        if machine != 0x8664:
            raise ValueError("not an x86-64 COFF object")
        strtab = symptr + 18 * nsym

        def name(raw):
            if raw[:4] == b"\0\0\0\0":
                off = struct.unpack_from("<I", raw, 4)[0]
                return data[strtab + off:data.index(b"\0", strtab + off)].decode(errors="replace")
            return raw.rstrip(b"\0").decode(errors="replace")

        self.sections = []          # 1-based: (raw bytes, {offset: (type, symbol index)})
        self.section_info = []      # 1-based: (name, characteristics)
        for k in range(nsec):
            h = 20 + opt + 40 * k
            sname, _, _, size, raw, relptr, _, nrel, _, chars = struct.unpack_from("<8sIIIIIIHHI", data, h)
            relocs = {}
            for r in range(nrel):
                va, si, typ = struct.unpack_from("<IIH", data, relptr + 10 * r)
                relocs[va] = (typ, si)
            if sname.startswith(b"/"):
                sname_s = name(b"\0\0\0\0" + struct.pack("<I", int(sname[1:].rstrip(b"\0") or 0)))
            else:
                sname_s = sname.rstrip(b"\0").decode(errors="replace")
            self.sections.append((data[raw:raw + size] if chars & 0x80 == 0 else b"", relocs))
            self.section_info.append((sname_s, chars))
        self.symbols = {}           # index: (section number, value)
        self.symbol_info = {}       # index: (name, storage class, type)
        k = 0
        while k < nsym:
            e = symptr + 18 * k
            value, secnum, typ, scl, naux = struct.unpack_from("<IhHBB", data, e + 8)
            self.symbols[k] = (secnum, value)
            self.symbol_info[k] = (name(data[e:e + 8]), scl, typ)
            k += 1 + naux

    def section(self, number):
        return self.sections[number - 1]

    def alignment(self, number):
        v = (self.section_info[number - 1][1] >> 20) & 0xF
        return 1 << (v - 1) if v else 16


COFF_REL = {1: "IMAGE_REL_AMD64_ADDR64", 2: "IMAGE_REL_AMD64_ADDR32", 3: "IMAGE_REL_AMD64_ADDR32NB",
            4: "IMAGE_REL_AMD64_REL32", 5: "IMAGE_REL_AMD64_REL32_1", 6: "IMAGE_REL_AMD64_REL32_2",
            7: "IMAGE_REL_AMD64_REL32_3", 8: "IMAGE_REL_AMD64_REL32_4", 9: "IMAGE_REL_AMD64_REL32_5"}


def disassemble_raw(objdump, code, ranges):
    """[Insn] of the byte ranges [(start, end)] of a code section, decoded
    without symbols (objdump -b binary), at their section offsets"""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for start, end in ranges:
            if end <= start:
                continue
            bin_path = os.path.join(tmp, "s.bin")
            with open(bin_path, "wb") as fh:
                fh.write(code[start:end])
            text = run(objdump, "-D", "-b", "binary", "-m", "i386:x86-64", "-w",
                       f"--adjust-vma={start:#x}", bin_path)
            for line in text.splitlines():
                m = INSN.match(line)
                if not m:
                    continue
                insn_text = m.group(3)
                code_bytes = [int(b, 16) for b in m.group(2).split()]
                if not insn_text.strip() and out:
                    prev = out[-1]
                    prev.code += code_bytes; prev.size = len(prev.code)
                    continue
                out.append(Insn(int(m.group(1), 16), code_bytes, insn_text))
    return out


def decode_coff_section(objdump, coff, n, unit, key, syms):
    """the functions of code section n, each decoded from its own start up
    to the next; the bytes of jump tables the dispatch code proves to be in
    the section (MSVC's) are data, everything else is decoded (an address
    relocation inside an instruction stays part of it)"""
    code, relocs = coff.section(n)
    sname = coff.section_info[n - 1][0]
    by_offset = {}
    for k, (sn, v) in coff.symbols.items():
        if sn == n and coff.symbol_info[k][2] == 0x20:
            by_offset.setdefault(v, set()).add(coff.symbol_info[k][0])
    if not by_offset:
        # hand-written assembly: its labels are untyped symbols (compilers
        # mark functions with type 0x20 and also emit untyped $LN labels)
        for k, (sn, v) in coff.symbols.items():
            nm, scl, _ = coff.symbol_info[k]
            if sn == n and scl in (2, 3, 6) and nm != sname and not nm.startswith((".", "$")):
                by_offset.setdefault(v, set()).add(nm)
    if not by_offset:
        by_offset = {0: {f"{sname}#{n}"}}               # code without a function symbol
    starts = sorted(by_offset)
    data = set()                                        # (start, end) of proven tables
    for _ in range(16):
        cuts = sorted(set(starts) | {a for r in data for a in r} | {len(code)})
        ranges = [(lo, hi) for lo, hi in zip(cuts, cuts[1:])
                  if not any(d0 <= lo < d1 for d0, d1 in data)]
        insns = disassemble_raw(objdump, code, ranges)
        for insn in insns:
            for off in range(insn.addr, insn.addr + insn.size):
                if off in relocs:
                    typ, si = relocs[off]
                    insn.relocs.append((off, COFF_REL.get(typ, f"IMAGE_REL_AMD64_{typ}"), key(si), 0))
        funcs = []
        for j, addr in enumerate(starts):
            end = starts[j + 1] if j + 1 < len(starts) else len(code)
            names = sorted(by_offset[addr])
            f = Function(unit, sname, names[0], addr, names)
            f.section_number, f.end = n, end
            f.insns = [i for i in insns if addr <= i.addr < end and not any(d0 <= i.addr < d1 for d0, d1 in data)]
            funcs.append(f)
        found = set()
        incoming = incoming_targets(funcs, "coff-object", {unit: syms}).get((unit, n), set())
        for f in funcs:
            for i, insn in enumerate(f.insns):
                if is_jump(insn) and insn.ops.startswith("*%"):
                    d = jump_table_dispatch(f, i, incoming)
                    if d is None:
                        continue
                    r = coff_table(f, d[0], d[1], d[2], d[3], coff, with_range=True)
                    if r is not None and r[1][0] == n:
                        found.add(r[1][1:])
        if found <= data:
            return funcs
        data |= found
    return funcs


def parse_coff(objdump, path):
    """functions, symbol and section tables, static names and the
    unreferenced static functions of a COFF object, from its own tables"""
    coff = Coff(path)
    unit = path
    funcs, syms, secs, local = [], {}, {}, set()
    def key(si):
        """section symbols (named like their, often duplicated, sections)
        by name@section, all others by name"""
        nm = coff.symbol_info.get(si, ("?", 0, 0))[0]
        sn = coff.symbols.get(si, (0, 0))[0]
        if sn > 0 and nm == coff.section_info[sn - 1][0]:
            return f"{nm}@{sn}"
        return nm

    for k, (secnum, value) in coff.symbols.items():
        if secnum > 0:
            syms[key(k)] = (secnum, value)
        nm, scl, typ = coff.symbol_info[k]
        if scl in (3, 6):
            local.add(nm)
    for n in range(1, len(coff.sections) + 1):
        secs[n] = coff.alignment(n)
    referenced, whole = set(), set()
    for n, (code, relocs) in enumerate(coff.sections, 1):
        sname, chars = coff.section_info[n - 1]
        unwind = sname.startswith(UNWIND_DEBUG_SECTIONS)
        for off, (typ, si) in relocs.items():
            nm, scl, styp = coff.symbol_info.get(si, ("", 0, 0))
            if not unwind:
                referenced.add(nm)
                if styp != 0x20 and coff.symbols.get(si, (0, 0))[0] > 0 \
                        and coff.section_info[coff.symbols[si][0] - 1][1] & 0x20:
                    whole.add(coff.symbols[si][0])     # a code section symbol: not resolved
        if not chars & 0x20 or not code:
            continue
        funcs += decode_coff_section(objdump, coff, n, unit, key, syms)
    # direct branches the assembler resolved (no relocation) refer too
    starts = {(f.section_number, f.addr): f for f in funcs}
    for f in funcs:
        for insn in f.insns:
            if insn.target is not None and not insn.relocs:
                t = starts.get((f.section_number, insn.target))
                if t is not None and t is not f:
                    referenced.update(t.aliases)
                    insn.target_name = insn.target_name or t.name
    dead = {(unit, f.name) for f in funcs
            if all(a in local and a not in referenced for a in f.aliases) and f.section_number not in whole}
    return coff, funcs, {unit: syms}, {unit: secs}, dead


UNWIND_DEBUG_SECTIONS = (".pdata", ".xdata", ".eh_frame", ".debug", ".gcc_except_table", ".sframe")


def unreferenced(objdump, path, funcs, local):
    """(unit, name) of the static functions of relocatable objects that no
    relocation and no direct branch refers to: copies a compiler emitted but
    nothing calls (MSVC keeps some of its inline functions so). A relocation
    against a code section symbol, whose target this does not resolve, keeps
    every function of that section."""
    refs, unit, whole, section = set(), path, set(), ""
    for line in run(objdump, "-r", "-w", path).splitlines():
        m = re.match(r"^(\S.*):\s+file format", line)
        if m:
            unit = m.group(1); continue
        m = re.match(r"^RELOCATION RECORDS FOR \[(.+)\]:", line)
        if m:
            section = m.group(1); continue
        if section.startswith(UNWIND_DEBUG_SECTIONS):
            continue          # unwind and debug data name every function
        m = re.match(r"^[0-9a-f]+\s+\S+\s+(\S+)$", line)
        if m:
            sym, _ = split_symbol(m.group(1))
            refs.add((unit, sym))
            if sym.startswith(".text"):
                whole.add((unit, sym))
    for f in funcs:
        for insn in f.insns:
            if insn.target_name:
                refs.add((f.unit, insn.target_name))
    dead = set()
    for f in funcs:
        if all(a in local.get(f.unit, set()) and (f.unit, a) not in refs for a in f.aliases) \
                and (f.unit, f.section) not in whole:
            dead.add((f.unit, f.name))
    return dead


# ---------------------------------------------------------------------------
# control flow

RET = {"ret", "retq", "retl", "retn"}
UNCOND = {"jmp", "jmpq", "ljmp"}


def is_jump(insn):
    """jumps, also the conditional loop family (loop, loope, loopne)"""
    return insn.mnem.startswith(("j", "loop"))


def branch_reloc(insn):
    """the symbol a branch's relocated displacement refers to: such a branch
    leaves the function (the assembler resolves branches within it), even
    where the unrelocated displacement points inside it"""
    if not (is_jump(insn) or insn.mnem.startswith("call")):
        return None
    for off, rtype, sym, addend in insn.relocs:
        if insn.addr < off < insn.addr + insn.size:
            return sym
    return None


def indirect(insn):
    return insn.target is None and insn.ops.startswith("*")


def successors(func, i, addr_index, blocks):
    insn = func.insns[i]
    m = insn.mnem
    nxt = [i + 1] if i + 1 < len(func.insns) else []
    if m in RET or m in ("ud2", "hlt"):
        return []
    if is_jump(insn):
        if branch_reloc(insn) is not None:
            return [] if m in UNCOND else nxt         # tail jump to another symbol
        if insn.target is not None:
            idx = addr_index.get(insn.target)
            if idx is None:
                return [] if m in UNCOND else nxt     # leaves the function: tail jump
            return [idx] if m in UNCOND else [idx] + nxt
        # indirect jump (a jump table): to any block start
        return blocks
    return nxt


def index_by_addr(func):
    return {insn.addr: i for i, insn in enumerate(func.insns)}


def block_starts(func, addr_index):
    """direct jump targets and instructions after a jump or return; the
    targets of jump tables are among them (they are not reached by falling
    through from the code before them)"""
    starts = set()
    for i, insn in enumerate(func.insns):
        if is_jump(insn) and insn.target in addr_index:
            starts.add(addr_index[insn.target])
        if (is_jump(insn) or insn.mnem in RET or insn.mnem == "ud2") and i + 1 < len(func.insns):
            starts.add(i + 1)
    return sorted(starts) or list(range(len(func.insns)))


def dataflow(func, entry, transfer, join):
    """forward data flow over the instructions; returns the state before each"""
    addr_index = index_by_addr(func)
    blocks = block_starts(func, addr_index)
    before = [None] * len(func.insns)
    if not func.insns:
        return before
    before[0] = entry
    work = [0]
    while work:
        i = work.pop()
        out = transfer(func.insns[i], before[i])
        for s in successors(func, i, addr_index, blocks):
            new = out if before[s] is None else join(before[s], out)
            if new != before[s]:
                before[s] = new
                work.append(s)
    return before


# ---------------------------------------------------------------------------
# alignment

TOP = None
CALLER_SAVED = {"rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"}


def reg(name):
    return GPR_ALIAS.get(name.lstrip("%"))


def imm(text):
    v = int(text, 16) if text.lower().startswith(("0x", "-0x")) else int(text)
    return v


def stack_transfer(insn, state):
    st = dict(state)
    m, ops = insn.mnem, [o.strip() for o in split_operands(insn.ops)]

    def shift(r, d):
        if st.get(r) is not TOP and r in st:
            st[r] = frozenset((x + d) % 32 for x in st[r])

    if (PUSH.match(m) or POP.match(m) or m.startswith("leave")) and operand_16bit(insn):
        # pushw/popw/leavew (and their 0x66-prefixed forms) move rsp by 2:
        # what they write is not tracked (all of it, if that is not known)
        w = written(insn)
        if w is None:
            return {r: TOP for r in st}
        for r in w | {"rsp"}:
            st[r] = TOP
        return st
    if PUSH.match(m):
        shift("rsp", -8); return st
    if POP.match(m):
        shift("rsp", 8)
        if ops and reg(ops[0]):
            st[reg(ops[0])] = TOP
        return st
    if m.startswith("leave"):
        st["rsp"] = frozenset((x + 8) % 32 for x in st["rbp"]) if st.get("rbp") else TOP
        st["rbp"] = TOP
        return st
    if m.startswith("call"):
        for r in CALLER_SAVED:
            st[r] = TOP
        return st
    if len(ops) == 2 and reg(ops[1]):
        dst = reg(ops[1])
        if m in ("sub", "subq", "add", "addq") and ops[0].startswith("$"):
            d = imm(ops[0][1:])
            shift(dst, -d if m.startswith("sub") else d); return st
        if m in ("and", "andq") and ops[0].startswith("$") and ops[1].startswith("%r"):
            # e.g. and $-32: the low bits of the mask keep what is known
            low = imm(ops[0][1:]) & 31
            known = st[dst] if st.get(dst, TOP) is not TOP else range(32)
            st[dst] = frozenset(x & low for x in known)
            return st
        if m in ("mov", "movq") and ops[0].startswith("%r") and reg(ops[0]) and ops[1].startswith("%r"):
            st[dst] = st.get(reg(ops[0]), TOP); return st
        if m in ("lea", "leaq") and ops[1].startswith("%r"):
            mm = MEM.fullmatch(ops[0])
            if mm and mm.group(2) and not mm.group(3) and reg(mm.group(2)) in st \
                    and st[reg(mm.group(2))] is not TOP:
                d = imm(mm.group(1)) if mm.group(1) else 0
                st[dst] = frozenset((x + d) % 32 for x in st[reg(mm.group(2))]); return st
            st[dst] = TOP; return st
    # any other write to a tracked register forgets it; unknown effects
    # forget everything
    w = written(insn)
    if w is None:
        return {r: TOP for r in st}
    for r in w:
        st[r] = TOP
    return st


def operand_16bit(insn):
    """a 16-bit operand-size form: the 0x66 prefix, a w suffix, a 16-bit register"""
    k = 0
    while k < len(insn.code) and insn.code[k] in LEGACY_PREFIXES:
        if insn.code[k] == 0x66:
            return True
        k += 1
    if re.match(r"^(push|pop)f?w$", insn.mnem):
        return True
    return any(re.fullmatch(r"%([abcd]x|[sd]i|[bs]p|r\d+w)", o) for o in split_operands(insn.ops))


def stack_join(a, b):
    out = {}
    for r in set(a) | set(b):
        x, y = a.get(r, TOP), b.get(r, TOP)
        out[r] = TOP if x is TOP or y is TOP else x | y
    return out


def split_operands(ops):
    out, depth, cur = [], 0, ""
    for c in ops.split("#")[0]:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return [o.strip() for o in out]


def rip_target(insn, syms, secs, linked):
    """(alignment of the section, offset in it) of a RIP-relative operand, or None"""
    disp = imm(insn.mem[0]) if insn.mem[0] else 0
    nxt = insn.addr + insn.size
    if linked:
        return 1 << 12, nxt + disp
    for off, rtype, sym, addend in insn.relocs:
        if not insn.addr <= off < nxt:
            continue
        if sym not in syms:
            return None
        sec, value = syms[sym]
        align = secs.get(sec)
        if align is None:
            return None
        if rtype.startswith("IMAGE_REL_AMD64_REL32"):
            k = int(rtype[-1]) if rtype[-2] == "_" else 0
            # the field holds the addend; the linker adds S - (P + 4 + k)
            return align, value + disp + addend + nxt - (off + 4 + k)
        if rtype in ("R_X86_64_PC32", "R_X86_64_PLT32", "R_X86_64_GOTPCREL"):
            # S + A - P, relative to the end of the instruction
            return align, value + addend + nxt - off
        return None
    return None


# ---------------------------------------------------------------------------
# checks

def check_file(objdump, path, allowed_wide, check_transitions=True, map_file=None, map_objects="."):
    """(functions with instructions, findings, names of those functions)"""
    flags = run(objdump, "-f", path)
    if "bigobj" in flags or object_kind(path) == "bigobj":
        raise SystemExit(f"{path}: /bigobj COFF objects are not supported")
    linked = "EXEC_P" in flags or "DYNAMIC" in flags
    # a single COFF object (also one without relocations); archives hold several
    coff_object = not linked and ("pe-x86-64" in flags or "coff-x86-64" in flags) \
        and not flags.count("file format") > 1
    findings = []
    if coff_object:
        coff, funcs, syms, secs, dead = parse_coff(objdump, path)
        ctx = Context("coff-object", coff=coff)
        ctx.incoming = incoming_targets(funcs, ctx.kind, syms)
        findings += decode_findings(funcs)
    elif map_file:
        mapped, image_base = msvc_map(map_file, map_objects)
        funcs = decode_mapped(objdump, path, mapped)
        # direct calls name what they call, also outside the audited objects
        # (the runtime helpers of NO_SSE_HELPERS)
        names_at = {a: names[0] for a, names, _ in mapped}
        for f in funcs:
            for insn in f.insns:
                if insn.target is not None and insn.target_name is None:
                    insn.target_name = names_at.get(insn.target)
        syms, secs, _, _ = tables(objdump, path)
        ctx = Context("image", data=image_bytes(objdump, path, (".text", ".rdata", ".rodata")),
                      image_base=image_base)
        ctx.incoming = incoming_targets(funcs, ctx.kind)
        names_at_all = {a: names[0] for a, names, _ in mapped}

        def redecode(start, stop):
            out = decode_range(objdump, path, start, stop)
            for insn in out:
                if insn.target is not None and insn.target_name is None:
                    insn.target_name = names_at_all.get(insn.target)
            return out
        funcs, problems = prune_unreachable(funcs, ctx, redecode)
        findings += problems
        dead = set()
    else:
        funcs = parse(objdump, path)
        syms, secs, local, fsyms = tables(objdump, path)
        for k, f in enumerate(funcs):
            names = fsyms.get((f.unit, f.section, f.addr), [])
            if set(names) - {f.name}:
                g = Function(f.unit, f.section, f.name, f.addr, names)
                g.insns, g.end = f.insns, f.end
                funcs[k] = g
        if linked:
            ctx = Context("image", data=image_bytes(objdump, path, (".text", ".rdata", ".rodata")))
        else:
            ctx = Context("elf-object", drel=data_relocations(objdump, path))
        ctx.incoming = incoming_targets(funcs, ctx.kind, syms)
        findings += decode_findings(funcs)
        dead = set() if linked else unreferenced(objdump, path, funcs, local)
    by_name = {}
    for f in funcs:
        for a in f.aliases:
            by_name.setdefault((f.unit, a), f)
        # branch targets: offsets within a section (sections start at 0 in objects)
        by_name.setdefault((f.unit, f.section, f.section_number, f.addr), f)

    for f in funcs:
        w = f.wide_isa
        for insn in f.insns:
            ext = extension(insn)
            if ext is None:
                continue
            if w and ext in allowed_wide:
                continue
            where = "in the 8-lane backend" if w else "outside the 8-lane backend"
            findings.append(("isa", f, insn, f"{ext} {where}"))

    for f in funcs:
        w = f.wide_flow
        needs = [(i, alignment_required(insn)) for i, insn in enumerate(f.insns)]
        needs = [(i, a) for i, a in needs if a and (w or a >= 32)]
        if not needs:
            continue
        entry = {"rsp": frozenset({8, 24})}
        before = dataflow(f, entry, stack_transfer, stack_join)
        usyms, usecs = syms.get(f.unit, {}), secs.get(f.unit, {})
        for i, a in needs:
            insn = f.insns[i]
            disp, mem_base, index, _ = insn.mem
            d = imm(disp) if disp else 0
            ok, why = False, ""
            if mem_base == "%rip":
                t = rip_target(insn, usyms, usecs, linked or map_file is not None)
                if t is None:
                    why = "RIP-relative target not resolved"
                else:
                    salign, off = t
                    ok = salign >= a and off % a == 0
                    why = f"RIP-relative offset {off:#x} in a 2**{salign.bit_length() - 1} section"
            elif mem_base and not index and before[i] is not None:
                st = before[i].get(reg(mem_base), TOP)
                if st is TOP:
                    why = f"{mem_base} not a tracked stack address"
                else:
                    ok = all((x + d) % a == 0 for x in st)
                    why = f"{mem_base}+{d} with {mem_base} = {sorted(st)} mod 32"
            else:
                why = "address not provable"
            if not ok:
                findings.append(("align", f, insn, f"{a}-byte aligned access: {why}"))

    if check_transitions:
        def local_jt(f, i):
            c = ctx
            if c.kind == "elf-object":
                c = Context("elf-object", syms=syms.get(f.unit, {}), drel=ctx.drel.get(f.unit, {}))
                c.incoming = ctx.incoming
            t = table_targets(f, i, c)
            return t is not None and all(f.addr <= a < f.span_end() for a in t)
        findings += transitions([f for f in funcs if (f.unit, f.name) not in dead], by_name, local_jt)
    # the number of functions with instructions, so that output this parser
    # does not understand cannot pass as a clean result
    names = {a for f in funcs if f.insns for a in f.aliases}
    return sum(1 for f in funcs if f.insns), findings, names


def transitions(funcs, by_name, local_jt=lambda f, i: False):
    wfuncs = [f for f in funcs if f.wide_flow]
    if not wfuncs:
        return []
    key = lambda f: (f.unit, f.name)
    # code that also has a baseline name is reached from baseline code
    callers = {key(f): (set() if f.wide_isa else {False}) for f in wfuncs}
    for f in funcs:
        for insn in f.insns:
            if insn.mnem.startswith("call") or is_jump(insn):
                t = callee(f, insn, by_name)
                if t is not None and t.wide_flow:
                    callers[key(t)].add(f.wide_flow)
    # entry functions are reached from other code (or from nowhere we can see)
    entry_state = {k: ({"C"} if (False in c or not c) else set()) for k, c in callers.items()}
    exit_state = {k: set() for k in callers}
    findings = {}
    addrs = {key(f): index_by_addr(f) for f in wfuncs}

    for _ in range(50):
        changed = False
        for f in wfuncs:
            k = key(f)
            st0 = frozenset(entry_state[k]) or frozenset({"C"})

            def transfer(insn, s, f=f):
                m = insn.mnem
                if m in ("vzeroupper", "vzeroall"):
                    return frozenset({"C"})
                if insn.enc in ("vex", "evex") and YMM.search(insn.ops):
                    return frozenset({"D"})
                if m.startswith("call"):
                    t = callee(f, insn, by_name)
                    if t is not None and t.wide_flow:
                        return frozenset(exit_state[key(t)]) or s
                return s

            before = dataflow(f, st0, transfer, lambda a, b: a | b)
            for i, insn in enumerate(f.insns):
                s = before[i]
                if s is None:
                    continue
                m = insn.mnem
                dirty = "D" in s
                if insn.enc == "legacy" and XMM.search(insn.ops) and dirty:
                    findings[(k, insn.addr)] = ("transition", f, insn,
                                                "legacy SSE instruction while the upper ymm state may be dirty")
                # also conditional: the taken edge leaves, the fall-through stays
                tail = is_jump(insn) and not indirect(insn) and (
                    branch_reloc(insn) is not None
                    or (insn.target is not None and insn.target not in addrs[k]))
                if is_jump(insn) and indirect(insn) and dirty and not local_jt(f, i):
                    # a jump table or a tail call (e.g. through the import
                    # table): not provably local, so the state must be clean
                    findings[(k, insn.addr)] = ("transition", f, insn,
                                                "indirect jump while the upper ymm state may be dirty")
                if m.startswith("call") or tail:
                    t = callee(f, insn, by_name)
                    if t is not None and t.wide_flow:
                        new = entry_state[key(t)] | s
                        if new != entry_state[key(t)]:
                            entry_state[key(t)] = set(new); changed = True
                        if tail:
                            ex = exit_state[k] | exit_state[key(t)]
                            if ex != exit_state[k]:
                                exit_state[k] = ex; changed = True
                    elif dirty and call_symbol(insn) not in NO_SSE_HELPERS:
                        what = "tail jump" if tail else "call"
                        findings[(k, insn.addr)] = ("transition", f, insn,
                                                    f"{what} to non-8-lane code while the upper ymm state may be dirty")
                if m in RET:
                    if dirty and (False in callers[k] or not callers[k]):
                        findings[(k, insn.addr)] = ("transition", f, insn,
                                                    "return to non-8-lane code while the upper ymm state may be dirty")
                    ex = exit_state[k] | s
                    if ex != exit_state[k]:
                        exit_state[k] = set(ex); changed = True
        if not changed:
            break
    return list(findings.values())


# Calls these may make with a dirty upper state: runtime helpers that
# execute no SSE instructions (stack probes, the stack protector and its
# failure path, assertion failures, which do not return).
NO_SSE_HELPERS = {"__stack_chk_fail", "__stack_chk_fail_local", "__chkstk", "___chkstk_ms",
                  "__security_check_cookie", "__report_gsfailure", "__assert_fail",
                  "_assert", "_wassert", "abort"}


def call_symbol(insn):
    for off, rtype, sym, addend in insn.relocs:
        return sym[:-4] if sym.endswith("@plt") else sym
    if insn.target_name:
        return insn.target_name
    t = re.search(r"<([^>+@]+)(@plt)?(\+0x[0-9a-f]+)?>", insn.ops)
    return t.group(1) if t and not t.group(3) else None


def callee(f, insn, by_name):
    """the called Function for a direct call or jump, if known"""
    for off, rtype, sym, addend in insn.relocs:
        name = sym[len(".text."):] if sym.startswith(".text.") else sym
        t = by_name.get((f.unit, name))
        if t is not None:
            return t
    if insn.target_name:
        return by_name.get((f.unit, insn.target_name))
    if insn.target is not None and not insn.relocs:
        t = by_name.get((f.unit, f.section, f.section_number, insn.target))
        if t is not None and t is not f:
            return t
    return None


# ---------------------------------------------------------------------------

def report(path, checked, findings):
    if checked == 0:
        print(f"{path}: no instructions disassembled")
        return 1
    if findings:
        print(f"{path}: {len(findings)} findings:")
        for code, f, insn, msg in findings[:60]:
            print(f"  {code}: {f.name}+{insn.addr - f.addr:#x}: {msg}: {insn.text.split('#')[0].strip()}")
        return 1
    print(f"{path}: OK ({checked} functions)")
    return 0


def compare(label, expect, result, entries=()):
    """result: check_file()'s; entries: functions that must have been audited
    (so that a fixture without findings cannot pass by auditing nothing)"""
    checked, findings, names = result
    missing = [e for e in entries if e not in names]
    if checked == 0 or missing:
        print(f"FAIL {label}: {checked} functions audited, missing {missing}")
        return 1
    got = sorted((code, f.name) for code, f, _, _ in findings)
    if got == expect:
        print(f"ok   {label}: {got if got else 'no findings'}")
        return 0
    print(f"FAIL {label}: expected {expect}, got {got}")
    for code, f, insn, msg in findings:
        print(f"    {code}: {f.name}: {msg}: {insn.text.split('#')[0].strip()}")
    return 1


def self_test(objdump, cc, cc_args, allowed, linker=None):
    """assemble the fixtures in tests/isa_audit and compare the findings
    with each fixture's '// expect:' lines (code and function); with linker,
    also link each COFF fixture into a DLL and audit that by its map"""
    fixdir = os.path.join(HERE, "isa_audit")
    status, n, empty_map_checked = 0, 0, False
    with tempfile.TemporaryDirectory() as tmp:
        # object headers: only MSVC /GL and LLVM bitcode objects are skipped
        anon = b"\x00\x00\xff\xff\x01\x00\x64\x86" + b"\x00" * 4
        for head, kind in ((anon + IL_MAGIC, "il"), (anon[:4] + b"\x02\x00" + anon[6:] + BIGOBJ_MAGIC, "bigobj"),
                           (anon + bytes(16), "other"), (b"BC\xc0\xde" + bytes(28), "bitcode"),
                           (b"\xde\xc0\x17\x0b" + bytes(28), "bitcode"), (b"\x64\x86" + bytes(30), "other")):
            p = os.path.join(tmp, "head.obj")
            with open(p, "wb") as fh:
                fh.write(head + bytes(64))
            got = object_kind(p)
            if got != kind:
                print(f"FAIL object header: expected {kind}, got {got}"); status = 1
        print("ok   object headers")
        for name in sorted(os.listdir(fixdir)):
            if not name.endswith(".S"):
                continue
            n += 1
            src = os.path.join(fixdir, name)
            obj = os.path.join(tmp, name[:-2] + ".obj")
            subprocess.run([*cc, *cc_args, "-c", src, "-o", obj], check=True)
            # "expect-coff:" lines add findings for PE/COFF objects only;
            # "expect-image:" lines, if any, replace all for the linked DLL
            coff = "pe-" in run(objdump, "-f", obj) or "coff" in run(objdump, "-f", obj)
            lines = open(src).read().splitlines()

            def expected(prefixes):
                return sorted(tuple(l.split(":", 1)[1].split()) for l in lines
                              if l.startswith(prefixes) and l.split(":", 1)[1].strip() != "none")
            prefixes = ("// expect:", "// expect-coff:") if coff else ("// expect:",)
            entries = [l.split()[1] for l in lines if l.strip().startswith(".globl")]
            status |= compare(name, expected(prefixes), check_file(objdump, obj, allowed), entries)
            if linker and coff and not any(l.startswith("// no-image:") for l in lines):
                dll, mp = os.path.join(tmp, name[:-2] + ".dll"), os.path.join(tmp, name[:-2] + ".map")
                subprocess.run([linker, "/nologo", "/dll", "/noentry", "/nodefaultlib", "/opt:noref",
                                "/opt:noicf", "/force:unresolved", "/ignore:4006", "/ignore:4088",
                                f"/map:{mp}", f"/out:{dll}", obj], check=False, capture_output=True)
                if not os.path.exists(mp):
                    print(f"FAIL {name}: {linker} produced no map"); status = 1; continue
                image = expected(("// expect-image:",)) if any(
                    l.startswith("// expect-image:") for l in lines) else expected(prefixes)
                status |= compare(name + " (DLL)", image,
                                  check_file(objdump, dll, allowed, map_file=mp), entries)
                if not empty_map_checked:
                    # control: a map without functions audits nothing
                    empty = os.path.join(tmp, "empty.map")
                    with open(empty, "w") as fh:
                        fh.writelines(l for l in open(mp) if "Publics by Value" not in l
                                      and not re.match(r"^\s*[0-9a-f]{4}:[0-9a-f]{8}\s+\S+\s+[0-9a-f]{8,16}\s", l))
                    if check_file(objdump, dll, allowed, map_file=empty)[0] != 0:
                        print("FAIL empty map: functions audited"); status = 1
                    else:
                        print("ok   empty map: nothing audited (report() fails such a run)")
                    empty_map_checked = True
    if n == 0:
        print("no fixtures found"); status = 1
    return status


def main():
    args = sys.argv[1:]
    objdump, fma, cc, cc_args, selftest, files = "objdump", True, None, [], False, []
    transitions_ = True
    wide_tu = False
    skip_empty = False
    map_file, map_objects, required = None, ".", []
    required_unless_ltcg, ltcg_objects, all_il_ok, linker = [], [], False, None
    require_wide = False
    header = CPU_HEADER
    while args:
        a = args.pop(0)
        if a == "--objdump":
            objdump = args.pop(0)
        elif a == "--no-fma":
            fma = False
        elif a == "--no-transitions":
            transitions_ = False
        elif a == "--wide-tu":
            wide_tu = True
        elif a == "--skip-empty":
            skip_empty = True
        elif a == "--map":
            map_file = args.pop(0)
        elif a == "--map-objects":
            map_objects = args.pop(0)
        elif a == "--require":
            required = [x for x in args.pop(0).split(",") if x]
        elif a == "--require-unless-ltcg":
            required_unless_ltcg = [x for x in args.pop(0).split(",") if x]
        elif a == "--ltcg-objects":
            ltcg_objects = [x for x in args.pop(0).split(";") if x]
        elif a == "--require-wide":
            require_wide = True
        elif a == "--all-il-ok":
            all_il_ok = True
        elif a == "--linker":
            linker = args.pop(0)
        elif a == "--header":
            header = args.pop(0)
        elif a == "--self-test":
            selftest = True
        elif a == "--cc":
            cc = [args.pop(0)]
        elif a == "--cc-arg":
            cc_args.append(args.pop(0))
        else:
            # CMake passes $<TARGET_OBJECTS:...> as one ;-separated argument
            files.extend(x for x in a.split(";") if x)
    allowed = wide_features(header, fma, wide_tu)
    version = run(objdump, "--version").splitlines()
    if not version or "GNU objdump" not in version[0]:
        # llvm-objdump formats instructions and relocations differently
        print(f"{objdump}: not GNU objdump ({version[0] if version else 'no version'})")
        return 2
    if selftest:
        if not cc:
            print(__doc__); return 2
        return self_test(objdump, cc, cc_args, allowed, linker)
    if not files:
        print(__doc__); return 2
    ltcg = [p for p in ltcg_objects if object_kind(p) in ("il", "bitcode")]
    if required_unless_ltcg:
        if ltcg:
            print(f"link-time code generation ({os.path.basename(ltcg[0])}): "
                  f"not required: {', '.join(required_unless_ltcg)}")
        else:
            required = required + required_unless_ltcg
    if map_file and ltcg_objects and not ltcg:
        # without link-time code generation the library's code is only in its
        # own objects: audit those (other objects, such as the test's, are
        # not compiled for this contract)
        own = {os.path.basename(p).lower() for p in ltcg_objects}
        map_objects = lambda obj, own=own: obj.split(":")[-1].lower() in own
        print("no link-time code generation: auditing the functions of " + ", ".join(sorted(own)))
    status, checked_any, skipped = 0, False, 0
    for path in files:
        kind = object_kind(path)
        if skip_empty and kind in ("il", "bitcode"):
            print(f"{path}: {'MSVC /GL object' if kind == 'il' else 'LLVM bitcode'} (no machine code), skipped")
            skipped += 1
            continue
        checked, findings, names = check_file(objdump, path, allowed, transitions_, map_file, map_objects)
        if required:
            missing = [r for r in required if r not in names]
            if missing:
                print(f"{path}: required functions not audited: {', '.join(missing)}")
                status = 1
        if require_wide and not any(wide(n) for n in names):
            print(f"{path}: no 8-lane function audited")
            status = 1
        checked_any = checked_any or checked > 0
        status |= report(path, checked, findings)
    if not checked_any:
        if all_il_ok and skipped == len(files):
            print("no object holds machine code (link-time code generation): the linked image is audited")
        else:
            print("no file held machine code")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
