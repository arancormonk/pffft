#!/usr/bin/env python3
"""Compile the standalone driver and measure existing CMake builds serially.

Example: python3 benchmarks/perf_analysis/run_matrix.py \
  --build release=build_simd_analysis/release --build native=build_simd_analysis/native \
  --output build_simd_analysis/results/matrix.csv --cpu 2
No CMake settings or production sources are changed by this script.
Every invocation appends a line to <output>.invocations:
START END ROWS_BEFORE ROWS_AFTER id=HEX seed=N bin.LABEL.PRECISION=SHA256... [changed=LABEL.PRECISION,...]
An invocation killed before it wrote that line (SIGKILL, a crash) leaves <output>.invocation-open;
the next --resume moves its rows to <output>.discarded, records it with ROWS_AFTER = ROWS_BEFORE
and discarded=K, and measures them again. <output>.lock admits one invocation at a time.
"""
import argparse
import contextlib
import csv
import errno
import hashlib
import itertools
import os
from pathlib import Path
import random
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
try:
    import fcntl
except ImportError:  # Windows: no lock on the CSV
    fcntl = None

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["build", "precision", "backend", "kind", "operation", "n", "repeat",
           "iterations", "seconds", "ns_per_call", "checksum"]
# perf counts cover the whole driver process: setup, checks, calibration, every repeat
PERF_COLUMN_PREFIX = "proc:"
# an SMT sibling more busy than this during a sample rejects it (--idle-sibling)
SIBLING_BUSY_LIMIT = 0.02
# the order of every run before --seed existed
DEFAULT_SEED = 1729
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIME_PATTERN = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
# a build label or precision as it can stand in a bin.LABEL.PRECISION= token
NAME_PATTERN = r"[^\s=.#]+"
# START ROWS_BEFORE id seed bin...; a marker of an earlier runner or script has no id
MARKER_PATTERN = (TIME_PATTERN + r" [0-9]+ id=[0-9a-f]{16} seed=[0-9]+(?: bin\." + NAME_PATTERN + r"\."
                  + NAME_PATTERN + r"=[0-9a-f]{64})*\n?")


def proc_stat_text():
    return Path("/proc/stat").read_text()


def sibling_list_text(cpu):
    return Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()


def aggregate_snapshot(text):
    """(total, busy) jiffies of the aggregate line of /proc/stat text; guest time is already in
    user time."""
    fields = list(map(int, text.splitlines()[0].split()[1:9]))
    return sum(fields), sum(fields) - fields[3] - fields[4]


def cpu_snapshot():
    """Linux aggregate CPU accounting."""
    return aggregate_snapshot(proc_stat_text())


def busy_cpus(before, after, cpus=None):
    return (cpus or os.cpu_count() or 1) * (after[1] - before[1]) / max(1, after[0] - before[0])


def wait_for_quiet(limit, siblings, interval=0.1, read=None, sleep=None, clock=None, cpus=None):
    """Returns once every condition asked for held over one and the same `interval`: a host
    load of at most `limit` - 1 busy CPUs (not checked when `limit` is None) and no sibling of
    `siblings` more than SIBLING_BUSY_LIMIT busy. While waiting, a line on standard error at
    most every 10 seconds saying what for. One sleep per attempt, whatever is checked."""
    read, sleep, clock = read or proc_stat_text, sleep or time.sleep, clock or time.monotonic
    last_message = 0
    while True:
        before = read()
        sleep(interval)
        after = read()
        reasons = []
        if limit is not None:
            busy = busy_cpus(aggregate_snapshot(before), aggregate_snapshot(after), cpus)
            # Leave one CPU of the limit available for the benchmark itself.
            if busy > limit - 1:
                reasons.append(f"host load: {busy:.1f} CPUs busy")
        if siblings:
            loaded = busy_siblings(parse_proc_stat_cpus(before), parse_proc_stat_cpus(after), siblings)
            if loaded:
                load = ", ".join(f"CPU {cpu} {share:.1%}" for cpu, share in sorted(loaded.items()))
                reasons.append(f"SMT sibling: {load} busy")
        if not reasons:
            return
        now = clock()
        if now - last_message >= 10:
            print("Waiting for " + "; ".join(reasons), file=sys.stderr, flush=True)
            last_message = now


def self_test():
    """the parsers and the sibling decision on canned text; runs no benchmark"""
    import io
    assert csv_columns([]) == COLUMNS, "the header without --perf-events"
    assert csv_columns(["cycles:u", "instructions:u"]) == COLUMNS + ["proc:cycles:u", "proc:instructions:u"]

    perf_output = (
        "# started on Wed Nov 11 18:40:05 2026\n"
        "\n"
        "1330189,,cycles:u,1007711,100.00,,\n"
        "2194305626,,instructions:u,1007711,100.00,1.65,insn per cycle\n"
        "<not counted>,,branch-misses:u,0,0.00,,\n"
        "<not supported>,,op_cache_hit_miss.op_cache_miss:u,0,100.00,,\n"
        "46572257,,ex_ret_brn:u,17994361,71.00,,\n")
    events = ["cycles:u", "instructions:u", "branch-misses:u",
              "op_cache_hit_miss.op_cache_miss:u", "ex_ret_brn:u"]
    # a count perf scaled (counted 71% of the time: multiplexed) is not stored
    assert parse_perf_stat(perf_output, events) == (["1330189", "2194305626", "", "", ""],
                                                    {"ex_ret_brn:u": "71.00"})
    # the columns follow the order of --perf-events, not perf's
    assert parse_perf_stat(perf_output, ["ex_ret_brn:u", "cycles:u"]) == (["", "1330189"],
                                                                          {"ex_ret_brn:u": "71.00"})
    assert parse_perf_stat("5,,cycles:u,9,100,,\n", ["cycles:u"]) == (["5"], {})
    for name, text, wanted in [("missing event", perf_output, ["cycles:k"]),
                               ("malformed line", "1330189 cycles:u\n", ["cycles:u"]),
                               ("non-numeric count", "12a,,cycles:u,1,100.00,,\n", ["cycles:u"]),
                               ("no percentage", "12,,cycles:u,1\n", ["cycles:u"]),
                               ("malformed percentage", "12,,cycles:u,1,9x.5,,\n", ["cycles:u"]),
                               ("empty percentage", "12,,cycles:u,1,,,\n", ["cycles:u"])]:
        try:
            parse_perf_stat(text, wanted)
        except RuntimeError:
            continue
        raise AssertionError(name)

    assert parse_cpu_list("14,30\n") == [14, 30]
    assert parse_cpu_list("0-1\n") == [0, 1]
    assert parse_cpu_list("0-1,4") == [0, 1, 4]

    def stat(times):
        """/proc/stat text with these per-CPU (busy, idle) jiffies"""
        lines = ["cpu  9999 0 0 9999 0 0 0 0 0 0"]
        for cpu, (busy, idle) in sorted(times.items()):
            # busy split over user, system and steal; idle over idle and iowait
            lines.append(f"cpu{cpu} {busy - 2} 0 1 {idle - 1} 1 0 0 1 0 0")
        return "\n".join(lines + ["intr 1 2 3", "ctxt 4"]) + "\n"

    before = parse_proc_stat_cpus(stat({6: (1000, 5000), 22: (2000, 9000), 30: (10, 10)}))
    assert before[6] == (6000, 1000) and before[22] == (11000, 2000)
    # 100 jiffies on the sibling: 2 busy is not more than 2%, 3 is
    quiet = parse_proc_stat_cpus(stat({6: (1100, 5000), 22: (2002, 9098), 30: (10, 10)}))
    assert busy_siblings(before, quiet, [22]) == {}, "2 busy jiffies of 100 are not more than 2%"
    # the benchmark CPU is not among its siblings
    assert smt_siblings(22, "6,22\n", stat({6: (1, 1), 22: (1, 1)})) == [6]
    assert smt_siblings(6, "6\n", stat({6: (1, 1)})) == []
    try:
        smt_siblings(22, "6,22,30\n", stat({6: (1, 1), 22: (1, 1)}))
    except ValueError as error:
        assert "30" in str(error)
    else:
        raise AssertionError("a sibling without a /proc/stat line")
    # one wait for whatever was asked: host load, idle siblings or both, over the same interval
    def waited(limit, siblings, host, sibling):
        """(sleeps, stderr) of wait_for_quiet() on 4 CPUs whose host load and sibling 22 are busy
        by host[k] and sibling[k] (fractions) in interval k, then quiet"""
        state = {"reads": 0, "host": 0, "sibling": 0, "sleeps": 0}

        def read():
            interval = state["reads"] // 2
            if state["reads"] % 2:
                state["host"] += 100 * (host[interval] if interval < len(host) else 0)
                state["sibling"] += 100 * (sibling[interval] if interval < len(sibling) else 0)
            state["reads"] += 1
            total = 100 * (state["reads"] // 2)
            return (f"cpu  {state['host']:.0f} 0 0 {total - state['host']:.0f} 0 0 0 0 0 0\n"
                    f"cpu22 {state['sibling']:.0f} 0 0 {total - state['sibling']:.0f} 0 0 0 0 0 0\n")

        def sleep(seconds):
            assert seconds == 0.1
            state["sleeps"] += 1
        ticks = iter(range(1000, 2000, 4))
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            wait_for_quiet(limit, siblings, read=read, sleep=sleep, clock=lambda: next(ticks), cpus=4)
        return state["sleeps"], stderr.getvalue()
    # 4 CPUs, limit 2.5: at most 1.5 busy CPUs, a fraction of 0.375
    assert waited(2.5, [22], [0.1], [0]) == (1, ""), "both quiet: one interval, one sleep"
    assert waited(2.5, [], [0.25], [0.5]) == (1, ""), "host load only: the sibling not checked"
    assert waited(None, [22], [1.0], [0]) == (1, ""), "siblings only: the host not checked"
    assert waited(2.5, [], [1.0, 1.0], []) == (3, "Waiting for host load: 4.0 CPUs busy\n")
    assert waited(2.5, [22], [1.0, 1.0], []) == (3, "Waiting for host load: 4.0 CPUs busy\n")
    assert waited(None, [22], [], [0.5, 0.5]) == (3, "Waiting for SMT sibling: CPU 22 50.0% busy\n")
    assert waited(2.5, [22], [], [0.5, 0.5]) == (3, "Waiting for SMT sibling: CPU 22 50.0% busy\n")
    assert waited(2.5, [22], [1.0], [0.5]) == (
        2, "Waiting for host load: 4.0 CPUs busy; SMT sibling: CPU 22 50.0% busy\n"), "both busy"
    # five busy intervals at 4 seconds apart: a line at 1000 and at 1012
    assert waited(2.5, [22], [1.0] * 5, [])[1].count("Waiting") == 2
    loaded = parse_proc_stat_cpus(stat({6: (1100, 5000), 22: (2003, 9097), 30: (110, 10)}))
    assert busy_siblings(before, loaded, [22]) == {22: 0.03}
    assert busy_siblings(before, loaded, [22, 30]) == {22: 0.03, 30: 1.0}
    assert busy_siblings(before, before, [22]) == {}, "no time elapsed"

    cases = list(itertools.product(["float"], ["real", "complex"], ["forward", "backward"], [64, 256]))
    builds = [("base", "B"), ("new", "N")]

    def order(seed):
        return [(case, " ".join("".join(label[0] for _, (label, _) in runs[i:i + 2])
                                for i in range(0, len(runs), 2)))
                for case, runs in measurement_order(cases, builds, 3, seed)]
    # computed by the runner before --seed existed; a change of the default order fails here
    assert order(DEFAULT_SEED) == [
        (("float", "complex", "forward", 256), "nb bn bn"),
        (("float", "complex", "backward", 256), "bn nb nb"),
        (("float", "real", "backward", 64), "bn nb bn"),
        (("float", "real", "forward", 256), "bn bn bn"),
        (("float", "complex", "backward", 64), "bn bn nb"),
        (("float", "complex", "forward", 64), "bn nb nb"),
        (("float", "real", "backward", 256), "nb bn bn"),
        (("float", "real", "forward", 64), "bn nb bn")], "the default order"
    assert all(repeat == i // 2 for _, runs in measurement_order(cases, builds, 3, 5)
               for i, (repeat, _) in enumerate(runs))
    assert order(5) == order(5) and order(5) != order(6) and order(5) != order(DEFAULT_SEED)
    for text, wanted in [("0", 0), ("1729", 1729)]:
        assert seed_value(text) == wanted
    for text in ["-1", "1.5", "x", "", " 3", "+3"]:
        try:
            seed_value(text)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"seed {text!r}")

    for name in ["native4", "wide", "a_b-c", "float"]:
        assert name_error("build label", name) is None, name
    for name in ["", "a b", "a\tb", "a=b", "a.b", "a#b"]:
        assert "invocation record" in name_error("build label", name), name
    digests = ["1" * 64, "2" * 64, "3" * 64]
    assert binary_tokens(5, {("new", "float"): digests[0], ("base", "float"): digests[1],
                             ("base", "double"): digests[2]}) == [
        "seed=5", "bin.base.double=" + digests[2], "bin.base.float=" + digests[1],
        "bin.new.float=" + digests[0]], "the token order"
    assert format_time(1e9) == time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(1e9))
    assert re.fullmatch(TIME_PATTERN, format_time(time.time()))

    start = {("new", "float"): "1" * 64, ("base", "float"): "2" * 64, ("base", "double"): "3" * 64}
    assert changed_binaries(start, lambda label, precision: start[label, precision]) == []
    assert changed_binaries(start, lambda label, precision: (
        "4" * 64 if label == "new" else start[label, precision])) == ["new.float"]

    def unreadable(label, precision):
        if precision == "double":
            raise FileNotFoundError(precision)
        return "5" * 64 if label == "new" else start[label, precision]
    changed = changed_binaries(start, unreadable)
    assert changed == ["base.double", "new.float"], "sorted like the bin tokens"
    assert changed_token(changed) == "changed=base.double,new.float"
    # sorted by label, then precision: not as text, where "a-b" would come before "a."
    assert changed_binaries({("a-b", "float"): "", ("a", "float"): ""}, lambda *_: "x") == [
        "a.float", "a-b.float"]

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        small = scratch / "abc"
        small.write_bytes(b"abc")
        assert sha256_file(small) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        # the comparison on real files: one rewritten, one deleted
        files = {name: scratch / name for name in ["base.float", "new.float"]}
        for path in files.values():
            path.write_bytes(b"abc")
        start = {(name.split(".")[0], "float"): sha256_file(path) for name, path in files.items()}

        def on_disk(label, precision):
            return sha256_file(scratch / f"{label}.{precision}")
        assert changed_binaries(start, on_disk) == []
        files["new.float"].write_bytes(b"abd")
        assert changed_binaries(start, on_disk) == ["new.float"]
        files["base.float"].unlink()
        assert changed_binaries(start, on_disk) == ["base.float", "new.float"]
        tokens = ["id=" + "ab" * 8] + binary_tokens(5, {("base", "float"): digests[0],
                                                        ("new", "float"): digests[1]})
        output = scratch / "x.csv"
        record, marker = invocation_files(output)
        assert record.name == "x.csv.invocations" and marker.name == "x.csv.invocation-open"
        assert discarded_file(output).name == "x.csv.discarded"
        t = [1.8e9 + 60 * i for i in range(4)]
        T = [format_time(v) for v in t]
        joined = " ".join(tokens)
        assert re.fullmatch(MARKER_PATTERN, f"{T[0]} 0 {joined}\n")
        assert re.fullmatch(MARKER_PATTERN, f"{T[0]} 0 id={'ab' * 8} seed=5")
        for text in [f"{T[0]} 0\n", f"{T[0]} 0 {' '.join(tokens[1:])}\n", f"{T[0]} 0 id=ab seed=5\n",
                     f"{T[0]} 0 {joined}\n{T[0]} 0 {joined}\n"]:
            assert not re.fullmatch(MARKER_PATTERN, text), text

        def rows(stream, count):
            for _ in range(count):
                stream.write("row,1\n")
            stream.flush()

        def records():
            return record.read_text().splitlines()

        # a normal invocation: the marker while it runs, the record after it
        assert check_output(output, False) is None
        stream = create_csv(output)
        stream.write("header,h\n")
        stream.flush()
        opened = start_invocation(output, tokens, t[0])
        assert opened == [T[0], "0"] + tokens and record.read_text() == ""
        assert marker.read_text() == f"{T[0]} 0 {joined}\n"
        rows(stream, 2)
        stream.close()
        assert finish_invocation(output, opened, 2, t[1]) == T[1]
        assert records() == [f"{T[0]} {T[1]} 0 2 {joined}"] and not marker.exists()
        for create in [lambda: check_output(output, False), lambda: create_csv(output)]:
            try:
                create()
            except FileExistsError:
                pass
            else:
                raise AssertionError("a new CSV is created exclusively")

        # chained: the next one begins where the last ended; a binary changed meanwhile
        stream = check_output(output, True)
        resolve_marker(output, stream, t[2])
        opened = start_invocation(output, ["id=" + "cd" * 8, "seed=6"], t[2])
        assert opened == [T[2], "2", "id=" + "cd" * 8, "seed=6"]
        rows(stream, 3)
        stream.close()
        finish_invocation(output, opened, 5, t[3], [changed_token(["base.float", "new.float"])])
        assert records()[1:] == [f"{T[2]} {T[3]} 2 5 id={'cd' * 8} seed=6 changed=base.float,new.float"]
        chain = [line.split() for line in records()]
        assert chain[0][2] == "0" and chain[-1][3] == str(count_rows(output)) == "5"
        assert all(a[3] == b[2] for a, b in zip(chain, chain[1:])), "the records chain"

        # never after a line cut short
        whole = record.read_bytes()
        record.write_bytes(whole[:-1])
        start_invocation(output, tokens, t[3])
        try:
            finish_invocation(output, opened, 5, t[3])
        except RuntimeError as error:
            assert str(record) in str(error)
        else:
            raise AssertionError("a record after an incomplete line")
        assert record.read_bytes() == whole[:-1] and marker.exists()
        record.write_bytes(whole)
        marker.unlink()

        # one invocation at a time; the CSV's own lock for another name of it
        if fcntl is not None:
            held = lock_output(output)
            for name, call in [("lock file", lambda: lock_output(output)),
                               ("CSV", lambda: check_output(output, True))]:
                with output.open("a") as csv_held:
                    if name == "CSV":
                        fcntl.flock(csv_held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        call()
                    except RuntimeError as error:
                        assert "another invocation is" in str(error) and str(output) in str(error), name
                    else:
                        raise AssertionError(name)
            held.close()
            lock_output(output).close()

        # stale companion files without their CSV
        for stale in invocation_files(scratch / "y.csv") + (discarded_file(scratch / "y.csv"),):
            stale.write_text("")
            try:
                check_output(scratch / "y.csv", False)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"stale {stale.name}")
            assert not (scratch / "y.csv").exists()
            stale.unlink()

        # --resume of a CSV without a companion file: its own record, the true rows before
        lone = scratch / "z.csv"
        lone.write_text("header,h\nrow,1\n")
        stream = check_output(lone, True)
        resolve_marker(lone, stream, t[0])
        opened = start_invocation(lone, tokens, t[0])
        stream.close()
        finish_invocation(lone, opened, 1, t[1])
        assert invocation_files(lone)[0].read_text() == f"{T[0]} {T[1]} 1 1 {joined}\n"
    with tempfile.TemporaryDirectory() as scratch:
        self_test_main(Path(scratch))
    print("run_matrix.py self-test passed")


def self_test_main(scratch):
    """main() itself on fake binaries, subprocess.run replaced by a fake driver and compiler"""
    import io
    from types import SimpleNamespace
    from unittest import mock
    builds = {label: scratch / label for label in ["base", "new"]}
    for label, directory in builds.items():
        directory.mkdir()
        (directory / "hotloops_float").write_bytes(label.encode())
    hashes = {(label, "float"): sha256_file(directory / "hotloops_float")
              for label, directory in builds.items()}
    bins = binary_tokens(9, hashes)[1:]
    calls = []
    compiles = []
    actions = {}  # sample number: what happens while the fake driver runs it

    perf = {"missing": set(), "empty": set(), "scaled": {}, "events": []}
    rejecting = set()  # samples the fake driver rejects

    def fake_run(command, **kwargs):
        perf_file = None
        if command[0] == "perf":
            perf_file = Path(command[command.index("-o") + 1])
            perf["events"].append(command[command.index("-e") + 1])
            command = command[command.index("--") + 1:]
        if not command[0].endswith("hotloops_float"):
            # the compiler: writes the build's original driver
            target = Path(command[command.index("-o") + 1])
            compiles.append(target)
            target.write_bytes(target.parent.name.encode())
            return SimpleNamespace(returncode=0)
        calls.append(Path(command[0]).parent.name)
        if len(calls) in actions:
            actions[len(calls)]()
        if len(calls) in rejecting:
            return SimpleNamespace(stdout="", stderr="invalid arguments: fake rejection\n", returncode=1)
        if perf_file is not None and len(calls) not in perf["missing"]:
            lines = ["# started on Wed Nov 11 18:40:05 2026", ""]
            for event in perf["events"][-1].split(","):
                share = "71.00" if perf["scaled"].get(len(calls)) == event else "100.00"
                lines.append(f"{1000 + len(calls)},,{event},5000,{share},,")
            perf_file.write_text("" if len(calls) in perf["empty"] else "\n".join(lines) + "\n")
        kind, operation, n, repeats = command[1:5]
        stdout = "".join(f"float,fake,{kind},{operation},{n},{k},1000,0.000100000,100.000,0\n"
                         for k in range(int(repeats)))
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    def run(output, *extra, compile=False, now=None, killed=False):
        """(exception or None, stderr) of main() on the two fake builds; `killed` ends it as SIGKILL
        would, before its record is written"""
        argv = ["--build", f"base={builds['base']}", "--build", f"new={builds['new']}",
                "--precisions", "float", "--kinds", "complex", "--sizes", "64", "256", "--ops", "forward",
                "--interleave", "--seconds", "0.001", "--compiler", "fake-c++", "--output", str(output)]
        handler = signal.getsignal(signal.SIGTERM)
        calls.clear()
        compiles.clear()
        error = None
        with contextlib.ExitStack() as patches:
            patches.enter_context(mock.patch.object(subprocess, "run", fake_run))
            if now is not None:
                patches.enter_context(mock.patch.object(time, "time", lambda: now))
            if killed:
                patches.enter_context(mock.patch.object(sys.modules[__name__], "finish_invocation",
                                                        lambda *_, **__: format_time(time.time())))
            stderr = patches.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                main(argv + ([] if compile else ["--no-compile"]) + list(extra))
            except BaseException as caught:
                error = caught
            finally:
                signal.signal(signal.SIGTERM, handler)
                actions.clear()
        return error, stderr.getvalue()

    def records(output):
        return [line.split() for line in invocation_files(output)[0].read_text().splitlines()]

    def csv_rows(output):
        with output.open(newline="") as stream:
            return [(row["build"], row["n"], row["repeat"]) for row in csv.DictReader(stream)]

    def complete_pairs(output):
        rows = csv_rows(output)
        pairs = {}
        for build, n, repeat in rows:
            pairs.setdefault((n, repeat), []).append(build)
        return len(set(rows)) == len(rows) and all(sorted(b) == ["base", "new"] for b in pairs.values())

    def snapshot():
        return {path: path.read_bytes() for path in sorted(scratch.rglob("*")) if path.is_file()}

    def killed_row(label, n, repeat):
        return f"{label},float,fake,complex,forward,{n},{repeat},1000,0.000100000,100.000,0\n"

    failures = []

    def check(name, condition):
        """condition: a value, or a function, whose IndexError or OSError is a failure too"""
        try:
            if not (condition() if callable(condition) else condition):
                failures.append(name)
        except (IndexError, OSError):
            failures.append(name)

    def is_id(token):
        return re.fullmatch(r"id=[0-9a-f]{16}", token) is not None

    # a normal invocation
    output = scratch / "normal" / "m.csv"
    record, marker = invocation_files(output)
    cases = [("float", "complex", "forward", n) for n in [64, 256]]
    expected = [(label, str(repeat)) for _, runs in measurement_order(cases, list(builds.items()), 2, 5)
                for repeat, (label, _) in runs]
    error, _ = run(output, "--repeats", "2", "--seed", "5")
    check("normal: no error", error is None)
    check("normal: the seeded order", [(b, r) for b, _, r in csv_rows(output)] == expected
          and calls == [b for b, _ in expected])
    first = records(output)
    check("normal: the record", lambda: len(first) == 1 and first[0][2:4] == ["0", "8"] and is_id(first[0][4])
          and first[0][5:] == binary_tokens(5, hashes)
          and all(re.fullmatch(TIME_PATTERN, field) for field in first[0][:2]))
    log = [line for line in output.with_suffix(".log").read_text().splitlines()
           if line.startswith("# invocation")]
    check("normal: the log", lambda: log == [f"# invocation {first[0][0]} seed=5 rows_before=0",
                                               f"# invocation {first[0][1]} rows_after=8"])
    check("normal: no marker", not marker.exists())
    check("normal: no temporary file", sorted(path.name for path in output.parent.iterdir())
          == ["m.csv", "m.csv.invocations", "m.csv.lock", "m.log"])

    # a resume chains to it, under another id
    error, _ = run(output, "--repeats", "3", "--seed", "6", "--resume")
    check("resume: no error", error is None and count_rows(output) == 12 and complete_pairs(output))
    check("resume: the record", lambda: records(output)[1][2:4] == ["8", "12"]
          and is_id(records(output)[1][4])
          and records(output)[1][4] != first[0][4] and records(output)[1][5:] == binary_tokens(6, hashes))

    # the driver fails in the second sample: the first sample's row is recorded
    actions[2] = lambda: (_ for _ in ()).throw(RuntimeError("fake driver failure"))
    error, _ = run(output, "--repeats", "5", "--seed", "7", "--resume")
    check("exception: propagates", isinstance(error, RuntimeError) and "fake driver" in str(error))
    check("exception: the record", records(output)[-1][2:4] == ["12", "13"] and count_rows(output) == 13)
    check("exception: no marker", not marker.exists())

    # SIGTERM during the second sample
    actions[2] = lambda: signal.raise_signal(signal.SIGTERM)
    error, _ = run(output, "--repeats", "5", "--seed", "8", "--resume")
    check("SIGTERM: SystemExit", isinstance(error, SystemExit) and error.code == 128 + signal.SIGTERM)
    check("SIGTERM: the record", records(output)[-1][2:4] == ["13", "14"] and count_rows(output) == 14)
    check("SIGTERM: no marker", not marker.exists())

    # a binary rewritten during the invocation
    actions[2] = lambda: (builds["new"] / "hotloops_float").write_bytes(b"other")
    error, stderr = run(output, "--repeats", "5", "--seed", "9", "--resume")
    check("changed: non-zero end", isinstance(error, RuntimeError) and "new.float" in str(error)
          and "new.float" in stderr)
    check("changed: the record", records(output)[-1][-1] == "changed=new.float"
          and records(output)[-1][5:-1] == binary_tokens(9, hashes))
    (builds["new"] / "hotloops_float").write_bytes(b"new")

    # killed after writing its record, before removing its marker: the ids are equal
    saved = []
    actions[1] = lambda: saved.append(marker.read_text())
    error, _ = run(output, "--repeats", "6", "--seed", "10", "--resume")
    count, rows = len(records(output)), count_rows(output)
    marker.write_text(saved[0] if saved else "")
    error, _ = run(output, "--repeats", "6", "--seed", "11", "--resume")
    check("record and marker: no error", error is None and saved)
    check("record and marker: only the marker removed", len(records(output)) == count + 1
          and not any(field.startswith("discarded=") for field in records(output)[-1])
          and count_rows(output) == rows and not marker.exists()
          and not Path(str(output) + ".discarded").exists())

    # killed with rows written, the last line incomplete or not, or after the CSV was truncated
    for name, tail, moved in [("killed", "", 3), ("incomplete line", "new,float,fa", 4),
                              ("after truncating", None, 0)]:
        output = scratch / name.replace(" ", "_") / "m.csv"
        record, marker = invocation_files(output)
        discarded = Path(str(output) + ".discarded")
        run(output, "--repeats", "2", "--seed", "5")
        start = format_time(time.time() - 100)
        killed = [start, "8", "id=" + "ab" * 8, "seed=9"] + bins
        marker.write_text(" ".join(killed) + "\n")
        lines = ""
        if tail is not None:
            lines = killed_row("base", 64, 2) + killed_row("new", 64, 2) + killed_row("base", 256, 2) + tail
            with output.open("a") as stream:
                stream.write(lines)
        error, stderr = run(output, "--repeats", "3", "--seed", "10", "--resume")
        check(f"{name}: no error", error is None)
        done = records(output)
        check(f"{name}: the record of the killed invocation", lambda:
              len(done) == 3
              and done[1][:1] + done[1][2:] == [start, "8", "8"] + killed[2:] + [f"discarded={moved}"]
              and re.fullmatch(TIME_PATTERN, done[1][1]))
        check(f"{name}: measured again", lambda: done[2][2:4] == ["8", "12"] and count_rows(output) == 12
              and complete_pairs(output) and not marker.exists())
        text = discarded.read_text() if discarded.exists() else ""
        check(f"{name}: the discarded rows",
              lambda: text == f"# {start} id={'ab' * 8}: {moved} rows after row 8, discarded {done[1][1]}\n"
              + lines + ("\n" if tail else ""))

    # a marker that names no id (an earlier runner's), does not parse, or begins beyond the CSV
    output = scratch / "unresolved" / "m.csv"
    record, marker = invocation_files(output)
    run(output, "--repeats", "2", "--seed", "5")
    start = format_time(time.time() - 100)
    for name, text in [("no id", f"{start} 8 seed=9 {' '.join(bins)}\n"), ("unparsable", "garbage\n"),
                       ("beyond the CSV", f"{start} 9 id={'ab' * 8} seed=9\n")]:
        marker.write_text(text)
        before = snapshot()
        error, _ = run(output, "--repeats", "3", "--seed", "10", "--resume", compile=True)
        check(f"{name}: refused", isinstance(error, RuntimeError) and str(marker) in str(error))
        check(f"{name}: nothing changed, nothing compiled", snapshot() == before and compiles == [])
    marker.unlink(missing_ok=True)

    # two invocations of one second that wrote nothing are not taken for one
    output = scratch / "collision" / "m.csv"
    record, marker = invocation_files(output)
    second = time.time() - 100
    run(output, "--repeats", "2", "--seed", "5", now=second)
    run(output, "--repeats", "2", "--seed", "6", "--resume", now=second)
    last = records(output)[-1]
    run(output, "--repeats", "2", "--seed", "6", "--resume", now=second, killed=True)
    opened = marker.read_text().split() if marker.exists() else ["", "", ""]
    check("collision: the same START, ROWS_BEFORE and tokens, another id",
          opened[:2] + opened[3:] == last[:1] + last[2:3] + last[5:] and opened[2] != last[4])
    error, _ = run(output, "--repeats", "2", "--seed", "6", "--resume", now=second + 5)
    check("collision: no error", error is None)
    check("collision: the killed one has its own record", lambda: records(output)[-2] == [
        last[0], format_time(second + 5), "8", "8", opened[2]] + last[5:] + ["discarded=0"])

    # killed through main() after writing rows: they are measured again by the next invocation
    output = scratch / "killed_main" / "m.csv"
    record, marker = invocation_files(output)
    run(output, "--repeats", "2", "--seed", "5")
    run(output, "--repeats", "4", "--seed", "6", "--resume", killed=True)
    opened = marker.read_text().split() if marker.exists() else ["", "", ""]
    written = csv_rows(output)[8:]
    error, _ = run(output, "--repeats", "4", "--seed", "7", "--resume")
    discarded = Path(str(output) + ".discarded")
    discarded = discarded.read_text().splitlines() if discarded.exists() else []
    check("killed in main(): the record", lambda: error is None
          and records(output)[1][:1] + records(output)[1][2:] == opened[:1] + ["8", "8"] + opened[2:]
          + ["discarded=8"])
    check("killed in main(): the discarded rows", len(discarded) == 9 and discarded[0].startswith(
        f"# {opened[0]} {opened[2]}: 8 rows after row 8, discarded ")
          and [tuple(line.split(",")[i] for i in (0, 5, 6)) for line in discarded[1:]] == written)
    check("killed in main(): measured again", lambda: records(output)[2][2:4] == ["8", "16"]
          and count_rows(output) == 16 and complete_pairs(output))

    # another process uses the output: nothing compiled, nothing changed
    if fcntl is not None:
        with open(str(output) + ".lock", "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for extra in [["--resume"], []]:
                before = snapshot()
                error, _ = run(output, "--repeats", "3", *extra, compile=True)
                check(f"lock {extra}: refused", isinstance(error, RuntimeError)
                      and f"another invocation is using {output}" in str(error))
                check(f"lock {extra}: nothing changed, nothing compiled",
                      snapshot() == before and compiles == [])
        # the CSV's own lock
        with output.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            error, _ = run(output, "--repeats", "3", "--resume", compile=True)
        check("CSV lock: refused", isinstance(error, RuntimeError)
              and "another invocation is writing" in str(error)
              and compiles == [])

    # what must or must not exist is checked before compiling
    error, _ = run(output, "--repeats", "3", compile=True)
    check("existing CSV: refused before compiling", isinstance(error, FileExistsError) and compiles == [])
    missing = scratch / "missing" / "m.csv"
    error, _ = run(missing, "--repeats", "3", "--resume", compile=True)
    check("missing CSV: refused before compiling", isinstance(error, FileNotFoundError) and compiles == [])
    missing.parent.mkdir(exist_ok=True)
    invocation_files(missing)[0].write_text("")
    error, _ = run(missing, "--repeats", "3", compile=True)
    check("stale companion: refused before compiling", isinstance(error, RuntimeError) and compiles == []
          and not missing.exists())

    # a record is written whole or not at all: a failure before the rename leaves the companion
    # file and the marker as they were, and the next --resume discards that invocation's rows
    for name, target in [("record write fails", "fsync"), ("record rename fails", "replace")]:
        output = scratch / name.replace(" ", "_") / "m.csv"
        record, marker = invocation_files(output)
        run(output, "--repeats", "2", "--seed", "5")
        failing = []
        real = {"fsync": os.fsync, "replace": os.replace}

        def fail(*args, kind=target):
            if failing:
                raise OSError(errno.ENOSPC, f"fake {kind} failure")
            return real[kind](*args)
        actions[4] = lambda: failing.append(True)
        before = record.read_bytes()
        with mock.patch.object(os, target, fail):
            error, _ = run(output, "--repeats", "4", "--seed", "6", "--resume")
        opened = marker.read_text().split() if marker.exists() else ["", "", ""]
        check(f"{name}: the error", isinstance(error, OSError) and "fake" in str(error))
        check(f"{name}: companion file and marker as they were", record.read_bytes() == before
              and opened[1:2] == ["8"] and count_rows(output) == 16)
        check(f"{name}: no temporary file", sorted(path.name for path in output.parent.iterdir())
              == ["m.csv", "m.csv.invocation-open", "m.csv.invocations", "m.csv.lock", "m.log"])
        error, _ = run(output, "--repeats", "4", "--seed", "7", "--resume")
        check(f"{name}: its rows discarded", lambda: error is None and records(output)[1][4:5] == opened[2:3]
              and records(output)[1][2:4] == ["8", "8"] and records(output)[1][-1] == "discarded=8"
              and count_rows(output) == 16 and complete_pairs(output))

    # a companion file whose last line is cut short is resolved by hand
    output = scratch / "torn" / "m.csv"
    record, marker = invocation_files(output)
    run(output, "--repeats", "2", "--seed", "5")
    saved = []
    actions[1] = lambda: saved.append(marker.read_text())
    run(output, "--repeats", "3", "--seed", "6", "--resume")
    whole = record.read_bytes()
    for name, torn, with_marker in [("torn record and its marker", whole[:-12], True),
                                    ("torn record", whole[:-1], False)]:
        record.write_bytes(torn)
        if with_marker:
            marker.write_text(saved[0] if saved else "")
        before = snapshot()
        error, _ = run(output, "--repeats", "4", "--seed", "7", "--resume", compile=True)
        check(f"{name}: refused", isinstance(error, RuntimeError) and str(record) in str(error))
        check(f"{name}: nothing changed, nothing compiled", snapshot() == before and compiles == [])
        marker.unlink(missing_ok=True)
    record.write_bytes(whole)

    # one name per file: no symbolic link (dangling or not) and no second hard link
    output = scratch / "links" / "m.csv"
    record, marker = invocation_files(output)
    run(output, "--repeats", "2", "--seed", "5")
    other = output.with_name("other")
    other.write_text("")
    for name, path, resume, make in [
            ("output a symbolic link", output.with_name("link.csv"), True,
             lambda path: path.symlink_to(output)),
            ("output a dangling symbolic link", output.with_name("dangling.csv"), False,
             lambda path: path.symlink_to(output.with_name("absent.csv"))),
            ("companion a symbolic link", record, True, None), ("marker a symbolic link", marker, True, None),
            ("discarded a symbolic link", discarded_file(output), True, None),
            ("CSV with another hard link", output.with_name("hard.csv"), True,
             lambda path: os.link(output, path)),
            ("output a second hard link", output.with_name("second.csv"), True,
             lambda path: os.link(output, path)),
            ("companion with another hard link", output.with_name("hard.invocations"), True,
             lambda path: os.link(record, path))]:
        target = output
        saved_record = None
        if make is None:
            if path.exists():
                saved_record = path.read_bytes()
                path.unlink()
            path.symlink_to(other)
        else:
            make(path)
            if name.startswith("output"):
                target = path
        before = snapshot()
        error, _ = run(target, "--repeats", "3", "--seed", "6", *(["--resume"] if resume else []),
                       compile=True)
        named = str(path if name.startswith(("output", "companion a", "marker", "discarded")) else
                    (output if name.startswith("CSV") else record))
        check(f"{name}: refused", isinstance(error, (RuntimeError, OSError)) and named in str(error))
        check(f"{name}: nothing changed, nothing compiled", snapshot() == before and compiles == [])
        path.unlink()
        if saved_record is not None:
            path.write_bytes(saved_record)
    error, _ = run(output, "--repeats", "3", "--seed", "6", "--resume")
    check("links: the output by its one name", error is None and count_rows(output) == 12)

    # one line per row: a quoted line break is refused before anything changes
    output = scratch / "multiline" / "m.csv"
    record, marker = invocation_files(output)
    run(output, "--repeats", "2", "--seed", "5")
    text = output.read_text().splitlines(keepends=True)
    text[1] = text[1].replace(",fake,", ',"fa\nke",')
    output.write_text("".join(text))
    marker.write_text(" ".join([format_time(time.time() - 100), "8", "id=" + "ab" * 8, "seed=9"] + bins)
                      + "\n")
    for name in ["multi-line field", "multi-line field without a marker"]:
        if name.endswith("marker"):
            marker.unlink(missing_ok=True)
        before = snapshot()
        error, _ = run(output, "--repeats", "3", "--seed", "6", "--resume", compile=True)
        check(f"{name}: refused", isinstance(error, RuntimeError) and str(output) in str(error))
        check(f"{name}: nothing changed, nothing compiled", snapshot() == before and compiles == [])
    try:
        count_rows(output)
    except RuntimeError as caught:
        check("multi-line field: count_rows", str(output) in str(caught))
    else:
        check("multi-line field: count_rows", False)
    # a directory link above the output, then "..": the file the kernel names, not a lexical one
    top = scratch / "spelling"
    (top / "real" / "sub").mkdir(parents=True)
    (top / "dirlink").symlink_to(top / "real" / "sub", target_is_directory=True)
    spelled = Path("dirlink") / ".." / "s0.csv"
    real, decoy = top / "real" / "s0.csv", top / "s0.csv"
    renames = []
    real_replace = os.replace

    def watched_replace(source, target):
        renames.append((source, target))
        return real_replace(source, target)
    working = os.getcwd()
    os.chdir(top)
    try:
        with mock.patch.object(os, "replace", watched_replace):
            error, _ = run(spelled, "--repeats", "2", "--seed", "5")
            check("spelling: an invocation", lambda: error is None and count_rows(real) == 8
                  and not decoy.exists() and len(records(real)) == 1)
            run(decoy, "--repeats", "2", "--seed", "5")
            before = {path: path.read_bytes() for path in top.iterdir() if path.is_file()}
            error, _ = run(spelled, "--repeats", "3", "--seed", "6", "--resume")
            check("spelling: a resume", lambda: error is None and count_rows(real) == 12
                  and records(real)[1][2:4] == ["8", "12"])
            run(spelled, "--repeats", "4", "--seed", "7", "--resume", killed=True)
            error, _ = run(spelled, "--repeats", "4", "--seed", "8", "--resume")
            check("spelling: the killed invocation's marker resolved", lambda: error is None
                  and records(real)[2][2:4] == ["12", "12"] and records(real)[2][-1] == "discarded=4"
                  and records(real)[3][2:4] == ["12", "16"] and count_rows(real) == 16
                  and not invocation_files(real)[1].exists())
            check("spelling: the decoy unchanged", {path: path.read_bytes() for path in top.iterdir()
                                                    if path.is_file()} == before)
            check("spelling: temporary files beside their files",
                  renames and all(os.path.samefile(os.path.dirname(source), os.path.dirname(target))
                                  for source, target in renames))
            (top / "real" / "link.csv").symlink_to(real)
            error, _ = run(Path("dirlink") / ".." / "link.csv", "--repeats", "5", "--resume", compile=True)
            check("spelling: the file itself a link", isinstance(error, RuntimeError)
                  and "symbolic link" in str(error) and compiles == [])
    finally:
        os.chdir(working)

    # perf: a fresh file every sample, a scaled count not stored, event names stripped
    output = scratch / "perf" / "m.csv"
    record, marker = invocation_files(output)
    perf["scaled"] = {3: "instructions:u"}
    error, _ = run(output, "--repeats", "2", "--seed", "5", "--perf-events", " cycles:u, instructions:u")
    check("perf: no error", error is None)
    with output.open(newline="") as stream:
        table = list(csv.reader(stream))
    check("perf: the columns", table[0][-2:] == ["proc:cycles:u", "proc:instructions:u"])
    check("perf: the event names stripped",
          perf["events"] and set(perf["events"]) == {"cycles:u,instructions:u"})
    check("perf: the counts", lambda: [row[-2:] for row in table[1:]] == [
        [str(1000 + i), "" if i == 3 else str(1000 + i)] for i in range(1, 9)])
    scaled = [line for line in output.with_suffix(".log").read_text().splitlines() if "scaled" in line]
    check("perf: the scaled count in the log", len(scaled) == 1 and "instructions:u" in scaled[0]
          and "71.00" in scaled[0] and calls[2] in scaled[0])
    perf["scaled"] = {}
    for name, kind in [("no perf file", "missing"), ("an empty perf file", "empty")]:
        perf[kind] = {2}
        error, _ = run(output, "--repeats", "3", "--seed", "6", "--resume",
                       "--perf-events", "cycles:u,instructions:u")
        perf[kind] = set()
        check(f"perf: {name} is an error naming the sample", isinstance(error, RuntimeError)
              and "perf" in str(error) and calls[1:2] and calls[1] in str(error)
              and "hotloops_float" in str(error))
        check(f"perf: {name}, the record written", lambda: records(output)[-1][3] == str(count_rows(output)))
    for events in ["cycles:u, cycles:u", "cycles:u, ", " "]:
        error, stderr = run(output, "--repeats", "3", "--resume", "--perf-events", events)
        check(f"perf: {events!r} refused", isinstance(error, SystemExit) and error.code == 2
              and "--perf-events" in stderr)

    # an SMT sibling busy before a sample: wait for it, do not run and reject the sample
    output = scratch / "sibling" / "m.csv"
    readings = {"count": 0, "busy": 0}

    def fake_stat():
        """CPU 6 measures, its sibling 22 is busy while readings["busy"] lasts"""
        readings["count"] += 1
        sibling_busy = 50 if readings["busy"] > 0 else 0
        readings["busy"] -= 1
        readings.setdefault("sibling", 0)
        readings["sibling"] += sibling_busy
        total = 100 * readings["count"]
        idle = total - readings["sibling"]
        return ("cpu  0 0 0 0 0 0 0 0 0 0\n"
                f"cpu6 {total} 0 0 0 0 0 0 0 0 0\ncpu22 {readings['sibling']} 0 0 {idle} 0 0 0 0 0 0\n")
    ticks = iter(range(1000, 100000, 4))
    sleeps = []
    for name, extra, busy in [("--idle-sibling", [], 6),
                              ("--idle-sibling and --max-busy-cpus", ["--max-busy-cpus", "2"], 6),
                              ("both, all quiet", ["--max-busy-cpus", "2"], 0)]:
        readings.update(busy=busy)
        sleeps.clear()
        with mock.patch.object(sys.modules[__name__], "proc_stat_text", fake_stat), \
                mock.patch.object(sys.modules[__name__], "sibling_list_text", lambda cpu: "6,22\n"), \
                mock.patch.object(os, "sched_setaffinity", lambda *_: None), \
                mock.patch.object(time, "sleep", sleeps.append), \
                mock.patch.object(time, "monotonic", lambda: next(ticks)):
            error, stderr = run(output, "--repeats", "2", "--seed", "5", "--idle-sibling", "--cpu", "6",
                                *extra)
        log = output.with_suffix(".log").read_text() if output.exists() else ""
        check(f"{name}: no error", error is None)
        check(f"{name}: no sample rejected", "Rejected" not in log and len(calls) == 8)
        check(f"{name}: waiting lines", stderr.count("Waiting for SMT sibling") == (1 if busy else 0))
        if not busy:
            check(f"{name}: one sleep per sample", sleeps == [0.1] * 8)
        output.unlink(missing_ok=True)
        for path in invocation_files(output) + (output.with_suffix(".log"),):
            path.unlink(missing_ok=True)

    # a sample the driver rejects stops the invocation with the driver's own message
    output = scratch / "failing" / "m.csv"
    run(output, "--repeats", "2", "--seed", "5")
    rejecting.add(2)
    error, _ = run(output, "--repeats", "4", "--seed", "7", "--resume")
    rejecting.clear()
    check("driver failure: the driver's message", isinstance(error, RuntimeError)
          and "invalid arguments: fake rejection" in str(error) and calls[1:2] and calls[1] in str(error)
          and "exit status 1" in str(error))
    check("driver failure: the record", lambda: records(output)[-1][2:4] == ["8", "9"])

    # leftover temporary files of a killed record or marker write: exactly those names are removed
    output = scratch / "leftover" / "m.csv"
    run(output, "--repeats", "2", "--seed", "5")
    leftovers = ["m.csv.invocations.abcd_123.tmp", "m.csv.invocation-open.zz9yx8w7.tmp"]
    kept = ["m.csv.invocations.tmp", "m.csv.invocations.abcd_123.tmp.keep", "n.csv.invocations.abcd_123.tmp",
            "m.csv.abcd_123.tmp", "m.csv.invocations.abc.tmp", "m.csv.invocations.ABCD_123.tmp"]
    for name in leftovers + kept:
        (output.parent / name).write_text(name)
    error, stderr = run(output, "--repeats", "3", "--seed", "6", "--resume")
    check("leftover: no error", error is None)
    check("leftover: removed", not any((output.parent / name).exists() for name in leftovers)
          and all(name in stderr for name in leftovers))
    check("leftover: nothing else", lambda: all((output.parent / name).read_text() == name for name in kept))

    # a trailing "/" or "/." names a directory, which pathlib would drop: a usage error
    top = scratch / "trailing"
    (top / "real" / "sub").mkdir(parents=True)
    (top / "dirlink").symlink_to(top / "real" / "sub", target_is_directory=True)
    os.chdir(top)
    try:
        for spelling in ["s0.csv/", "s0.csv/.", "dirlink/../s0.csv/", "dirlink/../s0.csv/./"]:
            for extra in [[], ["--resume"]]:
                before = snapshot()
                error, stderr = run(spelling, "--repeats", "2", *extra, compile=True)
                check(f"{spelling!r} {extra}: refused", isinstance(error, SystemExit) and error.code == 2
                      and "--output" in stderr and spelling in stderr)
                check(f"{spelling!r} {extra}: nothing created, nothing compiled",
                      snapshot() == before and compiles == [])
        for spelling, extra in [("s0.csv", []), ("./s0.csv", ["--resume"]), ("dirlink/../t0.csv", []),
                                ("./dirlink/../t0.csv", ["--resume"])]:
            error, _ = run(spelling, "--repeats", "2" if not extra else "3", *extra)
            check(f"{spelling!r}: accepted", error is None)
        check("accepted paths: their files", lambda: count_rows(top / "s0.csv") == 12
              and count_rows(top / "real" / "t0.csv") == 12)
    finally:
        os.chdir(working)
    if failures:
        raise AssertionError("main(): " + "; ".join(failures))


def csv_columns(perf_events):
    return COLUMNS + [PERF_COLUMN_PREFIX + event for event in perf_events]


def parse_perf_stat(text, events):
    """(the counts in `perf stat -x,` output in the order of `events`, {event: percentage} of
    the events perf scaled).

    An event perf did not count (<not counted>, <not supported>) gives "". So does one perf
    counted for less than 100% of the time (multiplexed): its count is an estimate, not a count."""
    counts = {}
    scaled = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(",")
        # value, unit, event, running time, percentage of it counted[, metric, unit]
        if len(fields) < 5 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", fields[4]):
            raise RuntimeError(f"malformed perf output: {line!r}")
        value, event, share = fields[0], fields[2], fields[4]
        if value in ("<not counted>", "<not supported>"):
            value = ""
        else:
            try:
                float(value)
            except ValueError:
                raise RuntimeError(f"malformed perf count: {line!r}") from None
            if float(share) < 100:
                value = ""
                scaled[event] = share
        counts[event] = value
    missing = [event for event in events if event not in counts]
    if missing:
        raise RuntimeError(f"perf output lacks {', '.join(missing)}")
    return [counts[event] for event in events], {event: scaled[event] for event in events if event in scaled}


def parse_cpu_list(text):
    """CPUs of a sysfs list such as "14,30" or "0-1"."""
    cpus = set()
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return sorted(cpus)


def parse_proc_stat_cpus(text):
    """Per-CPU (total, busy) jiffies of /proc/stat text, as in cpu_snapshot()."""
    snapshot = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and fields[0] != "cpu":
            values = list(map(int, fields[1:9]))
            snapshot[int(fields[0][3:])] = (sum(values), sum(values) - values[3] - values[4])
    return snapshot


def smt_siblings(cpu, siblings_list, proc_stat):
    """The SMT siblings of `cpu` from its thread_siblings_list text, without `cpu` itself;
    a sibling without a line in the /proc/stat text is a ValueError."""
    siblings = [other for other in parse_cpu_list(siblings_list) if other != cpu]
    missing = set(siblings) - set(parse_proc_stat_cpus(proc_stat))
    if missing:
        raise ValueError(f"no /proc/stat line for CPU {sorted(missing)}")
    return siblings


def busy_siblings(before, after, siblings, limit=SIBLING_BUSY_LIMIT):
    """{cpu: busy fraction} of the siblings more than `limit` busy between two snapshots."""
    loaded = {}
    for cpu in siblings:
        total = after[cpu][0] - before[cpu][0]
        busy = (after[cpu][1] - before[cpu][1]) / max(1, total)
        if busy > limit:
            loaded[cpu] = busy
    return loaded


def seed_value(text):
    if not re.fullmatch(r"[0-9]+", text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a non-negative integer")
    return int(text)


def measurement_order(cases, builds, repeats, seed):
    """[(case, [(repeat, build)...])]: the cases shuffled, then the builds within each repeat."""
    rng = random.Random(seed)
    cases = list(cases)
    rng.shuffle(cases)
    order = []
    for case in cases:
        runs = []
        for repeat in range(repeats):
            builds_now = list(builds)
            rng.shuffle(builds_now)
            runs.extend((repeat, build) for build in builds_now)
        order.append((case, runs))
    return order


def name_error(kind, name):
    """Why a build label or precision cannot be written to an invocation record, or None."""
    if not re.fullmatch(NAME_PATTERN, name):
        return (f"{kind} {name!r} must be non-empty and contain no whitespace, '=', '.' or '#': "
                "it could not be written to the invocation record")
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_tokens(seed, hashes):
    """The tokens after ROWS_AFTER for {(label, precision): sha256}."""
    return [f"seed={seed}"] + [f"bin.{label}.{precision}={digest}"
                               for (label, precision), digest in sorted(hashes.items())]


def changed_binaries(hashes, digest):
    """LABEL.PRECISION of every binary whose digest(label, precision) is no longer the one in
    {(label, precision): sha256} taken at the start, or cannot be read, sorted like the bin tokens."""
    changed = []
    for label, precision in sorted(hashes):
        try:
            same = digest(label, precision) == hashes[label, precision]
        except OSError:
            same = False
        if not same:
            changed.append(f"{label}.{precision}")
    return changed


def changed_token(changed):
    return "changed=" + ",".join(changed)


def format_time(seconds):
    return time.strftime(TIME_FORMAT, time.localtime(seconds))


def count_rows(path):
    """Data rows of a CSV, header not counted. A record over more than one line (a quoted line
    break) is an error: the runner finds rows by lines and never writes one."""
    rows = line = 0
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            for row in reader:
                if reader.line_num != line + 1:
                    raise RuntimeError(f"{path}: the CSV record ending on line {reader.line_num} spans "
                                       "more than one line")
                line = reader.line_num
                rows += bool(row)
        except csv.Error as error:
            raise RuntimeError(f"{path}: {error}") from None
    return max(0, rows - 1)


def replace_file(path, data):
    """Writes `data` to a temporary file beside `path`, fsyncs it and renames it over `path`, so
    that a reader sees the whole old content or the whole new one; a failure leaves `path` as it was."""
    # mkstemp makes `dir` absolute with os.path.abspath, which collapses "link/.." lexically;
    # realpath resolves through the file system and names the same directory
    descriptor, temporary = tempfile.mkstemp(dir=os.path.realpath(path.parent), prefix=path.name + ".",
                                             suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp's 0600 would replace the file's mode
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, path.stat().st_mode & 0o7777 if path.exists() else 0o666 & ~umask)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def invocation_files(output):
    """(<output>.invocations, <output>.invocation-open)"""
    return Path(str(output) + ".invocations"), Path(str(output) + ".invocation-open")


def finish_invocation(output, opened, rows_after, end, extra=()):
    """Adds the record of the invocation `opened` ([START, ROWS_BEFORE, tokens...]) to the
    companion file, whole (replace_file), then removes its open marker; returns END as written."""
    record, marker = invocation_files(output)
    end = format_time(end)
    content = record.read_bytes() if os.path.lexists(record) else b""
    if content and not content.endswith(b"\n"):
        raise RuntimeError(f"{record}: its last line has no newline; not appending after it")
    line = " ".join([opened[0], end, opened[1], str(rows_after)] + list(opened[2:]) + list(extra))
    replace_file(record, content + (line + "\n").encode())
    marker.unlink()
    return end


def last_record(record):
    """The fields of a companion file's last record, or None."""
    if not record.exists():
        return None
    lines = [line.split("#", 1)[0].split() for line in record.read_text().splitlines()]
    lines = [fields for fields in lines if fields]
    return lines[-1] if lines else None


def lock_output(output):
    """<output>.lock, locked until it is closed: one invocation at a time uses an output."""
    # before a lock file is created for a second name
    refuse_other_names(output)
    if fcntl is None:
        return None
    path = str(output) + ".lock"
    try:
        # O_NOFOLLOW: a symbolic link would create or lock a file of another name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                             0o666)
    except OSError:
        if os.path.islink(path):
            raise RuntimeError(f"{path} is a symbolic link: the runner uses one name per file") from None
        raise
    stream = os.fdopen(descriptor, "a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"another invocation is using {output}") from None
    return stream


def lock_csv(stream, output):
    """The CSV's own lock, which a hard link under another name (another lock file) shares."""
    if fcntl is None:
        return
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"another invocation is writing {output}") from None


def discarded_file(output):
    return Path(str(output) + ".discarded")


def refuse_other_names(output):
    """One name per file: the marker, the lock and the discarded rows are found by the CSV's
    name, which a symbolic link (dangling or not) or a second hard link would change."""
    record, marker = invocation_files(output)
    for path in [output, record, marker, discarded_file(output), output.with_suffix(".log")]:
        if os.path.islink(path):
            raise RuntimeError(f"{path} is a symbolic link: the runner uses one name per file")
    for path in [output, record]:
        if os.path.lexists(path) and os.lstat(path).st_nlink > 1:
            raise RuntimeError(f"{path} has {os.lstat(path).st_nlink} hard links: the runner uses one "
                               "name per file")


def remove_leftover_temporaries(output):
    """Removes the temporary files replace_file() left for the companion file and the marker
    when an invocation was killed while writing them; under the lock no writer owns one."""
    for target in invocation_files(output):
        directory = os.path.realpath(target.parent)
        # mkstemp's names: the prefix, 8 characters of its alphabet, the suffix
        pattern = re.escape(target.name) + r"\.[a-z0-9_]{8}\.tmp"
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if re.fullmatch(pattern, name) and os.path.isfile(path) and not os.path.islink(path):
                os.unlink(path)
                print(f"Removed {path}, left by an invocation killed while writing it",
                      file=sys.stderr, flush=True)


def check_output(output, resume):
    """What must or must not exist, under lock_output() and before compiling. With `resume`,
    returns the existing CSV opened for appending and locked; otherwise None."""
    record, marker = invocation_files(output)
    # again under the lock
    refuse_other_names(output)
    if resume:
        # no O_CREAT: --resume continues an existing CSV
        stream = os.fdopen(os.open(output, os.O_WRONLY | os.O_APPEND), "a", newline="")
        try:
            lock_csv(stream, output)
            if os.path.lexists(record):
                content = record.read_bytes()
                if content and not content.endswith(b"\n"):
                    raise RuntimeError(f"{record}: its last line has no newline (a record cut short?): "
                                       "to be resolved by hand")
            count_rows(output)
        except BaseException:
            stream.close()
            raise
        return stream
    if os.path.lexists(output):
        raise FileExistsError(errno.EEXIST, "the CSV exists (--resume continues it)", str(output))
    # stale files of an earlier run must not describe new data
    stale = [str(path) for path in invocation_files(output) + (discarded_file(output),)
             if os.path.lexists(path)]
    if stale:
        raise RuntimeError(f"{', '.join(stale)} exists, but {output} does not")
    return None


def resolve_marker(output, stream, now):
    """The open marker of an invocation killed before it removed it, with the CSV `stream`
    from check_output(). If its record is there (the ids are equal) the marker goes; otherwise
    its rows are not kept: moved to <output>.discarded, the CSV truncated, a record with
    ROWS_AFTER = ROWS_BEFORE and discarded=K appended, the marker removed, in that order."""
    record, marker = invocation_files(output)
    if not marker.exists():
        return
    text = marker.read_text()
    if not re.fullmatch(MARKER_PATTERN, text):
        raise RuntimeError(f"{marker}: unparsable open invocation marker, or one without id= "
                           f"(an earlier runner's) {text!r}: to be resolved by hand")
    opened = text.split()
    last = last_record(record)
    if last is not None and last[4:5] == [opened[2]]:
        # killed after appending its record and before removing the marker
        marker.unlink()
        return
    rows_before = int(opened[1])
    rows = count_rows(output)
    if rows_before > rows:
        raise RuntimeError(f"{marker}: the invocation began at row {rows_before}, "
                           f"{output} has {rows} data rows")
    data = output.read_bytes()
    # the header and the first ROWS_BEFORE rows stay; the rest, an incomplete line included, goes
    offset, keep = 0, 1 + rows_before
    for line in data.splitlines(keepends=True):
        if keep == 0:
            break
        offset += len(line)
        keep -= bool(line.strip())
    moved = data[offset:]
    count = sum(1 for line in moved.splitlines() if line.strip())
    with discarded_file(output).open("ab") as discarded:
        discarded.write(f"# {opened[0]} {opened[2]}: {count} rows after row {rows_before}, "
                        f"discarded {format_time(now)}\n".encode())
        discarded.write(moved if moved.endswith(b"\n") or not moved else moved + b"\n")
        discarded.flush()
        os.fsync(discarded.fileno())
    os.ftruncate(stream.fileno(), offset)
    os.fsync(stream.fileno())
    finish_invocation(output, opened, rows_before, now, [f"discarded={count}"])
    print(f"Discarded {count} rows of the invocation {opened[2]} of {opened[0]}, killed before "
          f"it wrote its record: {discarded_file(output)}", file=sys.stderr, flush=True)


def create_csv(output):
    """A new CSV, created exclusively and locked, with its empty companion file."""
    # Exclusive creation prevents an accidental rerun from replacing evidence.
    stream = output.open("x", newline="")
    lock_csv(stream, output)
    invocation_files(output)[0].open("x").close()
    return stream


def start_invocation(output, tokens, start):
    """Writes the open marker of an invocation of the CSV check_output() or create_csv() opened;
    returns [START, ROWS_BEFORE, tokens...]."""
    marker = invocation_files(output)[1]
    opened = [format_time(start), str(count_rows(output))] + list(tokens)
    replace_file(marker, (" ".join(opened) + "\n").encode())
    return opened


@contextlib.contextmanager
def deferred_signals():
    """SIGINT and SIGTERM wait until the rows in the CSV and their count agree again."""
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def terminate(signum, frame):
    """SIGTERM unwinds like Ctrl-C, so that the invocation record is written."""
    raise SystemExit(128 + signum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", help="LABEL=CMAKE_BUILD_DIR (required)")
    # a string: Path() would drop a trailing "/" that the check below refuses
    parser.add_argument("--output", help="(required)")
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.05)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384, 65536])
    parser.add_argument("--ops", nargs="+", default=["forward", "backward", "ordered-forward",
                                                  "ordered-backward", "reorder", "zconv", "pipeline"])
    parser.add_argument("--kinds", nargs="+", default=["real", "complex"])
    parser.add_argument("--precisions", nargs="+", default=["float", "double"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--interleave", action="store_true",
                        help="Alternate builds between samples using fresh processes")
    parser.add_argument("--max-busy-cpus", type=float,
                        help="Linux: wait for quiet host, reject samples exceeding this total CPU load")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interleaved CSV with unchanged binaries and parameters; "
                        "the rows of an invocation killed before it wrote its record are moved to "
                        "<output>.discarded and measured again")
    parser.add_argument("--perf-events", metavar="EV[,EV...]",
                        help="Run each sample under perf stat and append a column "
                        + PERF_COLUMN_PREFIX + "<event> per event. The counts are for the whole "
                        "driver process (setup, checks, calibration and every repeat), not for "
                        "the timed interval alone")
    parser.add_argument("--idle-sibling", action="store_true",
                        help="Linux, with --cpu: reject and repeat a sample during which an SMT "
                        f"sibling of that CPU was more than {SIBLING_BUSY_LIMIT * 100:g}%% busy, and "
                        "before each attempt wait until the siblings were idle by that rule for 0.1 s")
    parser.add_argument("--seed", type=seed_value, default=DEFAULT_SEED,
                        help="Seed of the order of the cases and of the builds within each repeat "
                        f"(default {DEFAULT_SEED}, which repeats the order of earlier runs; "
                        "independent invocations, such as the parts of one comparison, need different seeds)")
    parser.add_argument("--self-test", action="store_true",
                        help="Test the parsers, the sibling decision, the order and the invocation "
                        "records on canned text and temporary files; runs no benchmark")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.build or args.output is None:
        parser.error("--build and --output are required")
    separators = "/" + (os.altsep or "") + ("" if os.sep == "/" else os.sep)
    if re.split("[" + re.escape(separators) + "]", args.output)[-1] in ("", ".", ".."):
        parser.error(f"--output {args.output!r} names a directory (a trailing separator, '.' or '..'), "
                     "not a CSV file")
    args.output = Path(args.output)
    if args.resume and not args.interleave:
        parser.error("--resume requires --interleave")
    if args.max_busy_cpus is not None and args.max_busy_cpus <= 1:
        parser.error("--max-busy-cpus must exceed 1")
    perf_events = []
    if args.perf_events is not None:
        perf_events = [event.strip() for event in args.perf_events.split(",")]
        if "" in perf_events or len(set(perf_events)) != len(perf_events):
            parser.error("--perf-events needs distinct, non-empty event names")
    columns = csv_columns(perf_events)
    siblings = []
    if args.idle_sibling:
        if args.cpu is None:
            parser.error("--idle-sibling requires --cpu")
        try:
            siblings = smt_siblings(args.cpu, sibling_list_text(args.cpu), proc_stat_text())
        except ValueError as error:
            parser.error(str(error))
        if not siblings:
            print(f"CPU {args.cpu} has no SMT sibling", file=sys.stderr, flush=True)
    builds = []
    for spec in args.build:
        if "=" not in spec:
            parser.error(f"--build {spec!r}: expected LABEL=CMAKE_BUILD_DIR")
        label, path = spec.split("=", 1)
        if name_error("build label", label):
            parser.error(name_error("build label", label))
        if label in [known for known, _ in builds]:
            parser.error(f"build label {label!r} given twice")
        builds.append((label, Path(path).resolve()))
    for precision in args.precisions:
        if name_error("precision", precision):
            parser.error(name_error("precision", precision))
    # absolute, the working directory prefixed and nothing collapsed: "link/.." is the link's
    # parent for the kernel, not the directory before "link" as os.path.abspath would have it;
    # not resolved either: a symbolic link to the file is refused, not followed
    args.output = args.output.absolute()
    signal.signal(signal.SIGTERM, terminate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # the lock, the checks and a killed invocation's marker before compiling, which rewrites drivers
    lock = lock_output(args.output)
    try:
        out = check_output(args.output, args.resume)
        remove_leftover_temporaries(args.output)
        try:
            if out is not None:
                resolve_marker(args.output, out, time.time())
            measure(args, builds, columns, perf_events, siblings, out)
        finally:
            if out is not None:
                out.close()
    finally:
        if lock is not None:
            lock.close()


def measure(args, builds, columns, perf_events, siblings, out):
    """Compiles, then measures into the CSV `out`, or into a new one when it is None."""
    for label, path in builds:
        if not args.no_compile:
            for precision in args.precisions:
                command = [args.compiler, "-O3", "-DNDEBUG", "-std=c++11", "-Wall", "-Wextra",
                           "-pedantic", "-Werror", "-DPFFFT_STATIC_DEFINE", "-DPFFASTCONV_STATIC_DEFINE",
                           "-I" + str(ROOT / "include"), str(Path(__file__).with_name("bench_hotloops.cpp"))]
                if precision == "double":
                    command += ["-DPERF_DOUBLE"]
                else:
                    command += [str(path / "libpffastconv.a")]
                command += [str(path / "libpffft.a"), "-lm", "-o", str(path / ("hotloops_" + precision))]
                subprocess.run(command, check=True)
    paths = dict(builds)

    def binary_digest(label, precision):
        return sha256_file(paths[label] / ("hotloops_" + precision))
    hashes = {(label, precision): binary_digest(label, precision)
              for label, _ in builds for precision in args.precisions}
    tokens = [f"id={secrets.token_hex(8)}"] + binary_tokens(args.seed, hashes)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = list(itertools.product(args.precisions, args.kinds, args.ops, args.sizes))
    cases = [(p, k, op, n) for p, k, op, n in cases
             if not (op == "zp" and k == "complex")
             and not (op.startswith("fastconv") and p == "double")
             and not (op == "fastconv-single" and k == "real")]
    order = measurement_order(cases, builds, args.repeats if args.interleave else 1, args.seed)
    completed = set()
    if args.resume:
        with args.output.open(newline="") as previous:
            reader = csv.DictReader(previous)
            if reader.fieldnames != columns:
                raise RuntimeError("Unexpected CSV header")
            for row in reader:
                if any(row.get(field) is None for field in columns):
                    raise RuntimeError("Incomplete CSV row")
                completed.add(tuple(row[field] for field in
                                    ["build", "precision", "kind", "operation", "n", "repeat"]))
    if out is None:
        # created after compiling: its drivers are not modified after the CSV's creation
        out = create_csv(args.output)
        csv.writer(out, lineterminator="\n").writerow(columns)
        out.flush()
    opened = start_invocation(args.output, tokens, time.time())
    rows_before = int(opened[1])
    written = 0
    changed = []
    mode = "a" if args.resume else "x"
    with out, contextlib.ExitStack() as stack:
        log = None
        try:
            log = stack.enter_context(args.output.with_suffix(".log").open(mode))
            log.write(f"# invocation {opened[0]} seed={args.seed} rows_before={rows_before}\n")
            log.flush()
            scratch = stack.enter_context(tempfile.TemporaryDirectory())
            writer = csv.writer(out, lineterminator="\n")
            perf_file = Path(scratch) / "perf_stat.csv"
            for index, ((precision, kind, operation, n), runs) in enumerate(order):
                for repeat, (label, path) in runs:
                    if (label, precision, kind, operation, str(n), str(repeat)) in completed:
                        continue
                    command = [str(path / ("hotloops_" + precision)), kind, operation, str(n),
                               str(1 if args.interleave else args.repeats), str(args.seconds)]
                    if perf_events:
                        command = ["perf", "stat", "-x,", "-e", ",".join(perf_events),
                                   "-o", str(perf_file), "--"] + command
                    while True:
                        # counts of an earlier attempt or sample must not be read for this one
                        perf_file.unlink(missing_ok=True)
                        if args.max_busy_cpus is not None or siblings:
                            # rather than run the sample and reject it, again and again
                            wait_for_quiet(args.max_busy_cpus, siblings)
                        if args.max_busy_cpus is not None:
                            before = cpu_snapshot()
                        if siblings:
                            siblings_before = parse_proc_stat_cpus(proc_stat_text())
                        result = subprocess.run(command, text=True, capture_output=True)
                        if result.returncode != 0:
                            # check=True would drop the driver's stderr, which says what was wrong
                            raise RuntimeError(f"the sample of {label} failed (exit status "
                                               f"{result.returncode}): {command}: {result.stderr.strip()}")
                        if args.max_busy_cpus is not None:
                            busy = busy_cpus(before, cpu_snapshot())
                            if busy > args.max_busy_cpus:
                                log.write(f"Rejected for host load ({busy:.2f} busy CPUs): {command}\n")
                                log.flush()
                                continue
                        if siblings:
                            loaded = busy_siblings(siblings_before, parse_proc_stat_cpus(proc_stat_text()),
                                                   siblings)
                            if loaded:
                                load = ", ".join(f"CPU {cpu} {share:.1%}"
                                                 for cpu, share in sorted(loaded.items()))
                                log.write(f"Rejected for SMT sibling load ({load} busy): {command}\n")
                                log.flush()
                                continue
                        break
                    counts = []
                    if perf_events:
                        if not perf_file.exists() or perf_file.stat().st_size == 0:
                            raise RuntimeError(f"perf wrote no counts for the sample of {label}: {command}")
                        counts, scaled = parse_perf_stat(perf_file.read_text(), perf_events)
                        for event, share in scaled.items():
                            log.write(f"perf scaled {event} (counted {share}% of the time), stored empty: "
                                      f"{label} {command}\n")
                    log.write(f"{label} {command}\n{result.stderr}")
                    rows = []
                    for row in csv.reader(result.stdout.splitlines()):
                        if len(row) != len(COLUMNS) - 1:
                            raise RuntimeError(f"malformed output: {row}")
                        if args.interleave:
                            row[5] = str(repeat)
                        rows.append([label] + row + counts)
                    # a sample's rows are written whole, and counted once flushed
                    with deferred_signals():
                        writer.writerows(rows)
                        out.flush()
                        written += len(rows)
                    log.flush()
                if index % 10 == 0 or index == len(order) - 1:
                    print(f"{index + 1}/{len(order)} cases complete", file=sys.stderr, flush=True)
        finally:
            with deferred_signals():
                # a binary replaced while the invocation ran must not be vouched for by its bin token
                changed = changed_binaries(hashes, binary_digest)
                end = finish_invocation(args.output, opened, rows_before + written, time.time(),
                                        [changed_token(changed)] if changed else [])
                if log is not None:
                    log.write(f"# invocation {end} rows_after={rows_before + written}\n")
                    log.flush()
                if changed:
                    print(f"Driver binaries changed during the invocation: {', '.join(changed)}",
                          file=sys.stderr, flush=True)
    # only here, so that an exception or signal already unwinding is not masked
    if changed:
        raise RuntimeError(f"driver binaries changed during the invocation: {', '.join(changed)}")


if __name__ == "__main__":
    main()
