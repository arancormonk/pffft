# PFFFT SIMD performance analysis on Ryzen 5950X

Analysis of revision `fb3abfe`, 18 October 2026. The default Release build
already vectorizes the FFT arithmetic. For the core FFT library, the useful
next steps are selecting the existing AVX double backend on capable x86 CPUs
and combining complex FFT finalization with reordering. For PFDSP consumers,
explicitly vectorizing the recursive mixer is the largest measured opportunity.

The original experiments are retained in [experiments](experiments) against
revision `fb3abfe`. The production implementation now specializes the complex
ordered path separately, vectorizes the recursive mixer, and fixes its unstable
rate domain. The measurements below describe the original investigation;
production validation is recorded at the end of this report.

## Configuration and measurement

The worktree had no existing build directory. The baseline is therefore a fresh
default CMake Release configuration of the current revision, compared with
`TARGET_C_ARCH=native` and `TARGET_CXX_ARCH=native`.

| Item | Configuration |
| --- | --- |
| Host | AMD Ryzen 9 5950X, x86-64, 16 cores / 32 threads |
| ISA | SSE2, AVX2 and FMA; no AVX-512 |
| Cache | 32 KiB L1 data and 512 KiB L2 per core; two 32 MiB L3 domains |
| Main compiler | GCC 16.2.1 20260810 |
| Compiler comparison | Clang 23.1.3 for the mixer |
| Release options | `-O3 -DNDEBUG`, C99 / C++11, warnings as errors, no fast math |
| Default FFT backend | SSE1 float; four doubles represented by two SSE2 registers |
| Native FFT backend | 128-bit SSE float with FMA; 256-bit AVX double with FMA |
| Unfused control | Native ISA, `PFFFT_DISABLE_FMA=ON`, `-ffp-contract=off` |
| Affinity | Initial measurements on CPU 2; confirmations on CPU 14 |
| Frequency policy | Performance governor, boost enabled; no frequency locking |

At the analyzed revision, the unfused comparison applies to **PFFFT**, not
PFDSP: its CMake FMA option was not propagated to PFDSP's mixer intrinsics.
The production implementation propagates that option to PFDSP as well.

`bench_hotloops.cpp` separates real/complex, float/double, forward/backward,
ordered/unordered, reordering, spectral multiplication and complete convolution
calls. Input is deterministic and bounded. Setup, allocation, reference checks
and output checksums are outside timed regions. Work buffers are supplied;
FFT timings use separate input/output buffers. Results are cache-reuse workloads,
not measurements of an application streaming from DRAM.

The initial matrix uses five samples with a 40 ms target per sample. Confirmation
samples alternate builds in randomized order, using fresh processes. Actual
duration, iteration count and checksum are retained in the CSVs. Statistics use
the median for each individual case, without averaging different FFT sizes.

Two FFT confirmation attempts encountered unrelated builds/tests saturating the
host and are excluded. The final runner waits for quiet conditions and rejects
runs above its aggregate CPU-load limit. It never stops other applications.
CPU 14 and its SMT sibling 30 share the second L3 domain; the initial CPU 2 and
sibling 18 share the first. Comparisons always use variants measured on the same
core in the same sweep.

The original mixer benchmark uses a 100 Hz process clock on this host. Its 10 ms
quantization is inadequate for small changes, so mixer results use
`bench_mixer.cpp` and `steady_clock`. It processes 32 blocks of complex samples,
at rate `-0.0009`, resets state/data between samples, checks finite and bounded
output, and reports ns per complex sample. Seven samples alternate all builds.

## Where execution time goes

Hardware cycle sampling of native builds, with no lost samples:

| Workload | Approximate cycle distribution |
| --- | --- |
| Float real FFT → multiply → inverse, N=4096 | `radf4_ps` 36%, `radb4_ps` 34%, finalize/preprocess 19%, `pffft_zconvolve` 7% |
| Double real FFT → multiply → inverse, N=4096 | `radf4_ps` 35%, `radb4_ps` 38%, finalize/preprocess 15%, `pffftd_zconvolve` 9% |
| Float complex ordered forward, N=4096 | `passf4_ps` 63%, transform wrapper 22%, complex finalize 14% |
| Float complex fast convolution, block=1024 | radix-4 stages 49%, `pffastconv_apply` 22%, finalize/preprocess 20%, spectral multiply 6% |

These are separate workload profiles, not percentages of a representative
application. GCC inlines some layout work into `pffft_transform_internal`.
At N=1536, the real forward profile also spends time in radix-2 and the inlined
radix-3 stage; a powers-of-two profile alone misses those paths.

The radix-4 loops in [pffft_priv_impl.h](../../src/pffft_priv_impl.h#L203) are
already explicitly vectorized. Replacing them with equivalent intrinsics would
not add SIMD. For the float real N=4096 pipeline, even doubling the speed of
spectral multiplication can improve total time by only about 4%, given its
measured share. This makes layout passes and compiler failures more useful
initial targets than another rewrite of the multiply kernel.

## Existing x86 backend selection

Initial Release versus native medians, ns per call:

| Workload | Default | Native | Speedup |
| --- | ---: | ---: | ---: |
| Float real forward, N=1024 | 510.5 | 440.8 | 1.16× |
| Double real forward, N=1024 | 933.3 | 448.4 | 2.08× |
| Float real convolution pipeline, N=1024 | 1103.9 | 928.6 | 1.19× |
| Double real convolution pipeline, N=1024 | 2067.1 | 1157.0 | 1.79× |
| Float real convolution pipeline, N=4096 | 5474.5 | 4697.7 | 1.17× |
| Double real convolution pipeline, N=4096 | 10564.8 | 7477.6 | 1.41× |

This comparison includes instruction selection, register width and CPU tuning;
it is not an FMA-only speedup. Float stays four-wide even with `-march=native`.
The native/unfused control shows much smaller changes, including regressions in
some FFT cases. Fewer arithmetic instructions alone are not a throughput result.

A separate spectral-kernel sweep isolates FMA on the same native ISA:

| Real spectral operation | Native unfused ns | Native FMA ns | Speedup |
| --- | ---: | ---: | ---: |
| Float accumulate, N=1024 | 111.9 | 94.7 | 1.18× |
| Double accumulate, N=1024 | 102.4 | 95.1 | 1.08× |
| Float accumulate, N=16384 | 1916.9 | 1719.2 | 1.12× |
| Double accumulate, N=16384 | 2778.6 | 2922.8 | 0.95× |

The zero-phase multiply control, which uses no FMA arithmetic, stays essentially
unchanged between those builds. FMA is worthwhile for the measured small float
accumulation loops, especially if an application performs many spectral
accumulations per transform. It is not a uniform improvement for every buffer
size or precision.

For a build dedicated to this host, the existing native options provide an
immediate benefit, especially for double. A distributed x86 library should use
separate baseline and AVX/FMA implementations with feature dispatch, or explicit
architecture-specific packages. `-march=native` permits instructions unavailable
on other CPUs; it is not a portable shipping baseline.
[GCC architecture options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

## Complex FFT layout fusion

The current complex ordered transform writes the final butterfly results,
then traverses them again to interleave and rearrange the spectrum. The inverse
does the corresponding reorder before preprocessing. The
[fusion prototype](experiments/fuse_complex_reorder.patch) uses the existing
`INTERLEAVE2` / `UNINTERLEAVE2` macros directly in complex finalize/preprocess
and adjusts buffer parity to remove the extra traversal.

This preserves the public spectrum layout and four-lane FFT representation.
It does not require AVX2, wider vectors, new alignment rules or a new public API.
The relevant code is [complex finalize/preprocess](../../src/pffft_priv_impl.h#L1478)
and [transform buffer selection](../../src/pffft_priv_impl.h#L1742).
The benefit applies to **complex ordered transforms**. PFFASTCONV uses
unordered real transforms, so this fusion does not accelerate that pipeline.

The load-screened confirmation contains **1344 accepted samples**: seven per
case across four builds, both precisions, four transform modes and six sizes
from 64 through 65536. Representative medians, ns per transform:

| Complex transform | Default before → fused | Speedup | Native before → fused | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Float forward, N=1024 | 1382.8 → 1229.0 | 1.13× | 1264.2 → 1107.0 | 1.14× |
| Float backward, N=1024 | 1394.1 → 1234.4 | 1.13× | 1252.3 → 1127.9 | 1.11× |
| Float forward, N=4096 | 6778.8 → 6329.8 | 1.07× | 6265.9 → 5766.3 | 1.09× |
| Float backward, N=4096 | 6800.7 → 6156.9 | 1.10× | 6187.5 → 5622.3 | 1.10× |
| Double forward, N=1024 | 2775.8 → 2530.6 | 1.10× | 1691.5 → 1416.3 | 1.19× |
| Double backward, N=1024 | 2794.5 → 2604.2 | 1.07× | 1632.6 → 1488.5 | 1.10× |
| Double forward, N=4096 | 13173.3 → 12710.2 | 1.04× | 8595.4 → 7991.7 | 1.08× |
| Double backward, N=4096 | 12913.8 → 12137.9 | 1.06× | 8087.8 → 7506.9 | 1.08× |

Before promoting the patch, specialize the ordered path cleanly and preserve
the performance of unordered calls. The prototype changes a common helper's
signature and branching. For example, native double unordered backward at
N=1024 has a 2.8% slower median, which needs investigation before merging.
The earlier independent native sweep also found consistent ordered-transform
improvements, with median absolute deviations below 0.13% in all its cases.

An additional 320-sample native comparison covers N=96, 480, 1536 and 5120,
both precisions, and real/complex ordered transforms. Float complex speedups
range from 1.03× to 1.14×. Double complex cases generally improve 1.06–1.09×,
but forward N=5120 is 2.2% slower. Real controls mostly remain within about 2%,
with float backward N=96 showing a 5.1% slower median. Keep the real path
separate and investigate these regressions during production implementation.

## Recursive mixer vectorization

[shift_recursive_osc_inp_c](../../src/pf_mixer.cpp#L1012) maintains eight
independent oscillator lanes, but GCC emits **56 scalar floating-point
instructions** in the loop and no packed arithmetic. The
[mixer prototype](experiments/vectorize_recursive_mixer.patch) uses two
four-lane vectors through the existing mixer macros, reducing that to **14
packed instructions** for the same eight samples. The two independent vector
recurrences also expose more parallel work than the existing four-lane routine.

Confirmation medians at block size 4096, ns per complex sample:

| Compiler and ISA | Existing eight-lane routine | Explicit SIMD prototype | Speedup |
| --- | ---: | ---: | ---: |
| GCC default Release | 1.1330 | 0.4215 | 2.69× |
| GCC native | 1.1559 | 0.3572 | 3.24× |
| Clang native | 0.3806 | 0.3619 | 1.05× |

The existing GCC native **four-lane SIMD** routine is 0.6817 ns/sample, so the
prototype is also 1.91× faster than that implementation. Its unchanged control
stays approximately 0.682 ns/sample in the prototype build. Clang already
vectorizes the original eight-lane code, explaining its much smaller gain.
The seven-sample min/max spread for the eight-lane N=4096 comparisons is below
1%. The initial CPU 2 sweep independently measured about 3.27× for GCC native.

This is a PFDSP mixer improvement, not a 3× speedup for FFTs or full applications.
The out-of-place mixer has a similar missed-vectorization pattern, but that
extension has not been benchmarked here.

**Numerical limitation:** the existing eight-step oscillator becomes badly
conditioned near an eight-sample phase advance of π. At `rate=0.125`, N=8192,
the original GCC native routine produces values near 2474 for inputs whose
components are at most one. The prototype inherits this defect. The measured
rate is `-0.0009`; numerical checks at small rates have worst absolute errors
below 5e-6, and block continuity is exact. Near the singular rate, ARM's
different contraction behavior also makes baseline/prototype divergence large.
Fix or explicitly handle this rate domain before deploying the optimized
routine as a generally reliable oscillator. The check program reports this
domain separately instead of treating finite output as numerical accuracy.

## Candidates that did not justify a change

| Candidate | Evidence and decision |
| --- | --- |
| Apply `TARGET_C_ARCH` to PFFASTCONV | The target currently omits architecture flags. Compiling just this file with `-march=native` made tested complex calls about 1–3% slower. Do not treat the missing flag as a proven performance bug. |
| Explicit complex channel packing | Assembly attributes much of `pffastconv_apply` to deinterleave loads and scalar interleave stores. A portable packing prototype changed complete-call time by roughly 0–3%, with no clear benefit at larger blocks. Extra reads/stores consume the instruction savings. |
| Always select the single-FFT complex convolution mode | It helped the tested 1024-sample case slightly, but lost at 4096. Select by workload, not globally. |
| Blanket AVX widening of float FFTs | The core assumes four-lane layouts and transposes. Eight lanes need a layout/codelet redesign or batching independent transforms. No measured speedup from such a redesign is claimed. |
| CIC integrators | Compiler reports and [source](../../src/pf_cic.cpp#L105) show loop-carried integrator dependencies and phase-indexed table loads. Naive loop vectorization is not an established win; a prefix-sum or multi-channel algorithm would need its own benchmark and arithmetic review. |

The rejected packing patch is retained for reproduction, not recommended for
merging. The first packing sweep overlapped an emulated correctness test, so
small differences are especially insufficient evidence of a win.

## Platform applicability and validation

| Platform | Existing backend | Consequence for these opportunities |
| --- | --- | --- |
| x86-64 | SSE float, SSE2 pairs or AVX double | Measured on this host. GCC mixer vectorization is the largest new loop win; AVX double selection is an existing build win. |
| ARM64 | NEON float, two NEON registers per four doubles | FFT fusion uses existing macros and passed ARM64 correctness tests under QEMU. Mixer prototype also compiled and passed the stated continuity/low-rate checks. Physical ARM64 performance still needs measurement. |
| ARMv7 | NEON float; scalar fallback for double | Shared FFT fusion is structurally applicable. Compiler load/store behavior needs native measurement. |
| PowerPC | AltiVec float / VSX double | Shared FFT layout macros make fusion applicable in principle. No performance result claimed. |
| WebAssembly | SIMD128 float and paired double | Fusion can remove a memory traversal without requiring relaxed SIMD. No browser/runtime performance result claimed. |
| Scalar fallback | Scalar or four-element scalar representation | Preserve this path; it is not evidence for vector hardware speedups. |

ARM64 already has NEON FMA paths in the FFT headers. The x86 SSE2→AVX width
opportunity therefore does not translate directly to ARM64. NEON arithmetic and
zip/unzip operations used here are documented in the
[Arm intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html).
QEMU timings are not used as performance evidence.

Validation completed:

- Default and native Release: **23/23 CTests** each.
- Native unfused, fusion native, fusion default and packing native: **21/21**
  each; these experimental configurations omit benchmark/example targets.
- ARM64 fusion: float and double accuracy tests through N=8192, **52 real and
  68 complex sizes** each, using an independent long-double reference. These
  include ordering identities, in-place transforms and scratch-buffer variants.
- ARM64 packing: **897 mode cases, zero failures**.
- Mixer: GCC baseline/prototypes, Clang baseline/prototype, ARM64 baseline/
  prototype; **30 continuity/finite-output cases** per build, with the low-rate
  numerical gate and singular-rate limitation described above.

## Reproduction

Run from the repository root, at revision `fb3abfe` to reproduce the original
tables. Linux `perf`, Python 3, CMake and a C++11 compiler
are sufficient for the host analysis. Choose an idle logical CPU and leave its
SMT sibling idle as well.

```sh
cmake -S . -B build_perf_release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_perf_release --parallel 8
ctest --test-dir build_perf_release --output-on-failure

cmake -S . -B build_perf_native -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_C_ARCH=native -DTARGET_CXX_ARCH=native \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_perf_native --parallel 8
ctest --test-dir build_perf_native --output-on-failure

python3 benchmarks/perf_analysis/run_matrix.py \
  --build release=build_perf_release --build native=build_perf_native \
  --output build_perf_matrix.csv --cpu 14 --interleave --max-busy-cpus 2
```

The runner compiles the standalone drivers against the specified libraries;
it does not reconfigure the libraries. Output files use exclusive creation so
accidental reruns cannot overwrite measurements. `--ops`, `--sizes`, `--kinds`
and `--precisions` restrict the matrix. Use `--no-compile` to reuse drivers.
`--seed` sets the order of the cases and of the builds within a repeat;
invocations that form one comparison need different seeds. Every invocation
appends a line to `<output>.invocations` with its seed and the SHA-256 of
each driver binary.

To isolate FMA, retain native ISA selection and disable both explicit FMA and
compiler contraction:

```sh
cmake -S . -B build_perf_unfused -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_C_ARCH=native -DTARGET_CXX_ARCH=native \
  -DPFFFT_DISABLE_FMA=ON -DCMAKE_C_FLAGS=-ffp-contract=off \
  -DCMAKE_CXX_FLAGS=-ffp-contract=off \
  -DPFFFT_BUILD_BENCHMARKS=OFF -DPFFFT_BUILD_EXAMPLES=OFF
cmake --build build_perf_unfused --parallel 8
python3 benchmarks/perf_analysis/run_matrix.py \
  --build native=build_perf_native --build unfused=build_perf_unfused \
  --output build_perf_spectral.csv --cpu 14 --interleave --max-busy-cpus 2 \
  --kinds real --ops zconv-scale zconv-acc zp --sizes 256 1024 16384 65536
```

Apply prototypes to an isolated source copy at the same revision, for example:

```sh
mkdir -p build_perf_fusion_src
git archive fb3abfe | tar -x -C build_perf_fusion_src
patch -d build_perf_fusion_src -p1 < \
  benchmarks/perf_analysis/experiments/fuse_complex_reorder.patch
cmake -S build_perf_fusion_src -B build_perf_fusion \
  -DCMAKE_BUILD_TYPE=Release -DTARGET_C_ARCH=native -DTARGET_CXX_ARCH=native \
  -DPFFFT_BUILD_BENCHMARKS=OFF -DPFFFT_BUILD_EXAMPLES=OFF
cmake --build build_perf_fusion --parallel 8
ctest --test-dir build_perf_fusion --output-on-failure
python3 benchmarks/perf_analysis/run_matrix.py \
  --build native=build_perf_native --build fusion=build_perf_fusion \
  --output build_perf_fusion.csv --cpu 14 --interleave --max-busy-cpus 2 \
  --kinds complex --ops ordered-forward ordered-backward forward backward
```

Use the same isolated-copy procedure for the mixer patch. Build PFDSP in each
configuration, then compile the drivers against the corresponding library:

```sh
c++ -O3 -std=c++11 -DPFDSP_STATIC_DEFINE -Iinclude \
  benchmarks/perf_analysis/bench_mixer.cpp build_perf_native/libpfdsp.a \
  -lm -o build_perf_native/hotloops_mixer
c++ -O3 -std=c++11 -DPFDSP_STATIC_DEFINE -Iinclude \
  benchmarks/perf_analysis/check_mixer.cpp build_perf_native/libpfdsp.a \
  -lm -o build_perf_native/check_mixer
build_perf_native/check_mixer
taskset -c 14 build_perf_native/hotloops_mixer recursive8 4096 7 0.1
```

`recursive4` is the existing SIMD control. Repeat with the prototype library,
and with Clang compiling PFDSP, to reproduce the compiler comparison.

For a cycle profile of the calibrated FFT driver:

```sh
perf record -e cycles:u -F 997 -o build_perf_pipeline.data -- \
  taskset -c 14 build_perf_native/hotloops_float real pipeline 4096 1 2
perf report -i build_perf_pipeline.data --stdio --no-children --sort symbol,dso
python3 benchmarks/perf_analysis/summarize.py build_perf_matrix.csv \
  --output build_perf_summary.csv
```

The original [evidence](results) includes 5534 raw samples in compressed CSVs,
[medians and spread](results/summary.csv) for 994 case groups, compiler commands,
host/library fingerprints, hardware profiles, disassembly and correctness logs.
[Environment metadata](results/environment.json)
records the rejected sweeps and the load-limit change used when resuming the
confirmation. `--resume` retains accepted samples and requires unchanged
binaries and benchmark parameters.

## Production implementation

The ordered and unordered complex layout helpers now share one implementation
header instantiated with compile-time layout selection. The unordered helper
instruction sequences match the original GCC native build. A separate complex
ordered transform manages buffer parity and removes the reorder pass. It is
kept out of line so real transforms avoid its stack-alignment prologue.

The recursive oscillator uses two four-lane vectors for eight samples on the
existing mixer SIMD backends, with a scalar fallback. Mixing in place, mixing
out of place, generation, and the four-lane mixer share the stable recurrence.
When `abs(k1) > 1`, the implementation uses `-1/k1` and `-k2` for a rotation
shifted by pi, then negates both state components. This bounds the intermediate
values near the previous singularities while preserving the public coefficient
and state layouts. Lane initialization uses double-precision rotations.

The new `test_pf_mixer` checks 300 cases through 65536 samples against independent
trigonometric references, including singular rates and nearby values, negative
rates, DC/Nyquist, unaligned buffers, exact block continuity, and rate changes.
Worst absolute error is below 0.00065 on the tested native and ARM64 builds;
the gate is 0.001 across all tested rates. The original library fails this test.

Release presets provide baseline, native, explicit Haswell, and native unfused
configurations. `PFFFT_DISABLE_FMA` now reaches PFDSP as well as PFFFT. CI tests
all four presets with GCC and Clang, and the mixer regression test joins the
existing ARM64, ARMv7, PowerPC, WebAssembly, Windows, and macOS matrix.

Local validation passed 24/24 tests for GCC baseline/native/Haswell/unfused,
Clang native, and both scalar configurations; AddressSanitizer passed 22/22
with benchmark/example targets disabled. ARM64 under QEMU passed the 300 mixer
cases and the float/double accuracy suites (52 real and 68 complex sizes each).

The [production results](results/production-summary.csv) contain 640 FFT
samples and 42 mixer samples; [metadata](results/production-environment.json)
records the library/source hashes and measurement conditions. Representative
native FFT medians (ns per call):

| Complex ordered transform | Original | Production | Speedup |
| --- | ---: | ---: | ---: |
| Float forward, N=1024 | 1266.4 | 1104.1 | 1.15× |
| Float backward, N=1024 | 1249.7 | 1125.7 | 1.11× |
| Float forward, N=4096 | 6242.1 | 5682.5 | 1.10× |
| Double forward, N=1024 | 1640.5 | 1424.4 | 1.15× |
| Double backward, N=4096 | 8035.8 | 7503.1 | 1.07× |
| Double forward, N=5120 | 11016.9 | 11279.0 | 0.98× |

The last case remains a 2.4% regression. Removing a memory pass does not improve
every size; the implementation keeps the shared portable path rather than
encoding an exception for one size on this host. Real/unordered control medians
range from 1.8% faster to 3.4% slower. The earlier 7.8% unordered-backward
regression from the first shared-helper refactor is gone, as is the stack
prologue imposed on real transforms by inlining the complex path.

At N=256 the production eight-lane mixer takes 0.4209 ns/sample with GCC baseline
(1.1358 originally, **2.70×**), 0.3577 with GCC native (1.1506, **3.22×**), and
0.3631 with Clang native (0.3932, **1.08×**). Larger production mixer timing cases
were deferred when unrelated workloads saturated the host; the original
prototype's larger-block results above are not relabelled as production data.

## Follow-up implementation

The table-driven mixer now computes the phasor for sample `i` from table entry
`i-1` in the same iteration. Handling sample zero separately removes the loop
dependency. The existing mixer SIMD macros process four samples at a time,
with a scalar fallback and tail. This avoids the inefficient interleaved
vector arithmetic that GCC generated for the baseline ISA. A shared helper
also preserves both components before writing, so `shift_unroll_cc`
supports identical input/output buffers. Table construction, phase advancement,
public signatures, and arbitrary sample counts are preserved.

Complex inverse transforms now interleave directly into the output buffer.
When the last FFT pass finishes in scratch, this combines the interleave with
the copy previously needed for in-place transforms. Sizes whose last pass
already finished in output retain the same interleave operation.

Ordered real transforms now fuse reordering into finalization/preprocessing.
Both layouts share the butterfly and DC/Nyquist arithmetic. The forward helper
carries the reversed quarters' boundary pairs between blocks, keeping its bulk
stores aligned; the inverse helper gathers the ordered data directly. In-place
transforms with an even FFT pass count retain the existing reorder path:
fusion would require a separate copy in those cases. The scalar-one
implementation also retains its existing layout handling.

The new table-mixer test covers 399 combinations of sizes, rates, and phases,
including unaligned buffers, vector tails, exact aliasing, canaries, zero-length
calls, and block continuity. The existing float/double FFT accuracy suites
check both directions against independent long-double references, require
bitwise ordered/unordered equivalence, and exercise in-place buffers and null
scratch. Local GCC baseline/native/Haswell/unfused, Clang native, and both scalar
builds passed 25/25 tests each; Debug AddressSanitizer passed 23/23.

The analysis drivers now accept `inplace-forward`, `inplace-backward`,
`inplace-ordered-forward`, and `inplace-ordered-backward` for FFTs, and
`table-inplace` and `table-out` for mixers. In-place FFT timing uses repeated
zero input to prevent overflow from repeated unnormalized transforms; random
data correctness checks run before timing. Mixer timing rotates 32 blocks at
rate -0.0009 cycles/sample; table creation is outside the timed region.

To compare these changes with `c21f0ba`, build that revision in an isolated
source tree and run the current drivers against both libraries:

```sh
python3 benchmarks/perf_analysis/run_matrix.py \
  --build before=build_before --build after=build_after \
  --output build_followup_real.csv --cpu 14 --interleave --max-busy-cpus 2.5 \
  --repeats 5 --seconds 0.04 --kinds real \
  --ops ordered-forward ordered-backward inplace-ordered-forward inplace-ordered-backward \
  --sizes 32 64 256 1024 1536 4096 5120 16384

c++ -O3 -std=c++11 -DPFDSP_STATIC_DEFINE -Iinclude \
  benchmarks/perf_analysis/bench_mixer.cpp build_after/libpfdsp.a \
  -lm -o build_after/bench_table
taskset -c 14 build_after/bench_table table-inplace 4096 7 0.08
```

The [follow-up results](results/followup-summary.csv) compare against `c21f0ba`
on a Ryzen 9 5950X. They include 1040 FFT samples, 168 mixer samples, and 18
additional control samples. The [metadata](results/followup-environment.json)
records source/library hashes, compiler commands, and measurement conditions;
the [validation log](results/followup-validation.log) records the local checks.
Each comparison uses randomized before/after processes pinned to CPU 14,
rejecting samples when aggregate host load exceeds 2.5 busy CPUs. Raw samples
are retained in the accompanying compressed CSV files.

Representative GCC native FFT medians (ns per call):

| Transform | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Float complex in-place backward, N=1024 | 1158.0 | 1037.6 | 1.12× |
| Double complex in-place backward, N=1024 | 1375.2 | 1283.3 | 1.07× |
| Float real in-place ordered forward, N=4096 | 2632.1 | 2321.4 | 1.13× |
| Float real in-place ordered backward, N=4096 | 2696.1 | 2318.6 | 1.16× |
| Double real in-place ordered forward, N=4096 | 4322.8 | 3707.1 | 1.17× |
| Double real in-place ordered backward, N=4096 | 4344.6 | 3694.0 | 1.18× |

Table-mixer medians at N=4096 (ns per complex sample):

| Configuration and mode | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| GCC baseline, in-place | 0.8710 | 0.3707 | 2.35× |
| GCC baseline, out-of-place | 0.9589 | 0.3628 | 2.64× |
| GCC native, in-place | 0.8656 | 0.3165 | 2.74× |
| GCC native, out-of-place | 0.8684 | 0.3194 | 2.72× |
| Clang native, in-place | 0.8680 | 0.3262 | 2.66× |
| Clang native, out-of-place | 0.8662 | 0.3252 | 2.66× |

At N=256 the table-mixer speedups range from 2.21× to 2.53×. FFT gains vary
with size and buffer layout. Two measured real-transform regressions remain:
double in-place ordered backward at N=32 increases from 33.4 to 36.7 ns
(9.9%), and double out-of-place ordered forward at N=1024 increases from
522.2 to 543.1 ns (4.0%). The full results retain these cases. An initially
noisy unordered double inverse control at N=1024 was repeated with nine
longer samples per build; its medians were 480.0 and 480.6 ns. Both sets of
control samples are retained. These timings describe this host; ARM64 under
QEMU passed the mixer tests and float/double FFT accuracy checks through
N=1024, but provides correctness evidence only.

## 8-lane AVX2 float backend and fused layout passes

The earlier analysis listed blanket AVX widening of float FFTs as needing a
layout and codelet redesign. That redesign is now implemented: an N-point
transform is split into 8 interleaved sub-sequences instead of 4, the
unchanged FFTPACK pass codelets run on `__m256` vectors, and new 8-lane
finalize/preprocess steps combine the lanes (8x8 transposes and a radix-8
butterfly; the real transform needs an 8-way real combination with its own
DC/Nyquist and sub-Nyquist boundary terms). It lives in
[pffft_wide_impl.h](../../src/pffft_wide_impl.h) and is selected per setup at
runtime, see the top-level [README](../../README.md#x86-8-lane-avx2-float-transforms).
Independently of the lane count, the complex transforms now deinterleave
their input inside the first FFTPACK pass and interleave the result inside
the last backward pass ([pffft_passes_impl.h](../../src/pffft_passes_impl.h)),
for all 4-lane backends (SSE, AVX double, NEON, AltiVec/VSX, WASM, scalar
vectors) as well.

### Why the width, not the radix

Controlled comparison: FFTW 3.3.10 built twice from the same tarball with the
same flags, `--enable-sse2` versus `--enable-sse2 --enable-avx --enable-avx2`,
both MEASURE, linked into the same `bench_pffft_float` (MFlops, average of a
forward and a backward call). Enabling FFTW's AVX/AVX2/FMA backend made its
complex float transforms 1.8-2.0x as fast at N = 64..4096, while at four
lanes FFTW (large-radix generated codelets, planner) was within 3-16% of the
original 4-lane PFFFT. A prototype that ran PFFFT's unchanged pass code on 8
lanes reproduced most of that gain, so the width was the lever. Literature on
the same short-vector formulation: Franchetti & Püschel, *Short Vector Code
Generation for the DFT* (IPDPS 2003); McFarlin et al., *Automatic SIMD
Vectorization of FFTs for the Larrabee and AVX Instruction Sets* (ICS 2011).

### Measurement

Ryzen 9 5950X (Zen 3), GCC 16.2.1, host-tuned Release builds, CPU 14, at most
2.5 busy CPUs per accepted sample, `run_matrix.py --interleave`, 5 samples per
case and build in randomized order (15 for the small-size confirmation).
Three builds of this revision's sources: `orig` is master 77a7ecd, `native4`
the final code with `PFFFT_ENABLE_WIDE_DISPATCH=OFF`, `wide` the final code.
Raw samples: [float](results/wide-final-float.csv.gz) (8910 samples, 27
sizes from 64 to 2^20, 11 operations), [double](results/wide-final-double.csv.gz),
[small sizes](results/wide-small-sizes.csv.gz), [fused-pass A/B](results/wide-b1-ab.csv.gz).

Median ns per call, `orig` → `wide`:

| Transform | N | orig | 4-lane final | 8-lane | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| complex forward | 256 | 222.3 | 206.7 | 127.0 | 1.75x |
| complex forward | 1024 | 1051.7 | 980.3 | 588.8 | 1.79x |
| complex forward | 4096 | 5453.6 | 4950.5 | 3445.2 | 1.58x |
| complex forward | 65536 | 127628.2 | 120331.9 | 89286.2 | 1.43x |
| complex ordered forward | 1024 | 1109.0 | 1036.0 | 653.1 | 1.70x |
| complex ordered backward | 1024 | 1112.1 | 1031.8 | 684.4 | 1.62x |
| complex FFT-multiply-IFFT | 1024 | 2322.5 | 2184.6 | 1342.1 | 1.73x |
| real forward | 256 | 108.0 | 108.7 | 82.5 | 1.31x |
| real forward | 1024 | 428.3 | 432.4 | 280.3 | 1.53x |
| real forward | 4096 | 2262.5 | 2303.1 | 1491.0 | 1.52x |
| real ordered forward | 1024 | 469.9 | 470.7 | 333.9 | 1.41x |
| real ordered backward | 1024 | 470.7 | 468.7 | 324.2 | 1.45x |
| real FFT-multiply-IFFT | 1024 | 962.1 | 943.3 | 631.1 | 1.52x |
| real FFT-multiply-IFFT | 65536 | 129513.0 | 128961.5 | 97206.0 | 1.33x |

N not divisible by 64 (complex) or 128 (real) keep the 4-lane code:
0.96-1.04x controls.

### Size table

Which sizes use the 8-lane code on a measured CPU is decided per transform
type and N by [wide_size_table.py](wide_size_table.py), which writes
[pffft_wide_sizes.h](../../src/pffft_wide_sizes.h). A size is enabled only
if, in each sweep of the CPU's group, all eight transform modes and the
FFT-multiply-IFFT pipeline were at least 1.10x as fast in the median with
non-overlapping interquartile ranges, and `pffft_zreorder` and
`pffft_zconvolve` were at least 0.97x. For Zen 3 these are the two sweeps on
the Ryzen 9 5950X described here and seven on EPYC 7763 runner VMs
([below](#measuring-another-cpu)). The two desktop sweeps
([1](results/wide-sweep1.csv.gz), [2](results/wide-sweep2.csv.gz), 26532
samples each) cover all 201 complex and 167 real sizes from 64 to 2^20 that
the 8-lane code supports, all eleven operations, 3 samples per case and
build in randomized interleaved order, with the `native4` and `wide` builds
above. The other settings are those of the measurement above, except that
the host-load limit was raised to 4 busy CPUs (sweep 1 started at 2.5) to
make progress on a shared host; the IQR condition guards against the extra
noise. Per-size verdicts: [wide-size-table.txt](results/wide-size-table.txt).

Enabled on Zen 3 desktop parts (`pffft_wide_zen3_desktop_*`, from these two
sweeps): 192 complex and 152 real sizes, where the gated modes measured
1.19-1.91x (complex, median 1.48x) and 1.12-1.97x (real, median 1.41x), and
no case of the eleven operations was below 0.97x. The Zen 3 server part
(EPYC 7763, seven runner sweeps, [below](#measuring-another-cpu)) passes
fewer: 143 and 115 sizes (`pffft_wide_zen3_server_*`), gated modes 1.10-1.94x
and 1.10-2.02x (medians 1.48x and 1.42x). Of the desktop sizes it fails 55
complex (N = 64, and all others from N = 62208, with working sets beyond its
512 KB L2 per core) and 47 real ones (N = 128, where real `pffft_zreorder`
measured 0.82x in its first three sweeps, and the rest from N = 80000); there
the 8-lane code was mostly still faster, but one mode fell under 1.10x or the
interquartile ranges overlapped, or reordering or spectral multiplication
fell under 0.97x (down to 0.78x: complex `pffft_zconvolve` at N = 288000).
So each model uses its own table, and the Zen 3 models that were not measured
(the Cezanne and Rembrandt APUs) the 137 complex and 105 real sizes that pass
in all nine sweeps (`pffft_wide_zen3_*`, exactly the sizes of both tables).
Spectral multiplication is up to 2x faster at small N and about equal once
it is memory bound.

Not enabled in the desktop sweeps (and so not on Zen 3 desktop parts and the
unmeasured Zen 3 models; the server table enables some of them): real
N = 256 (in-place backward 1.10x in both sweeps, just under the threshold),
and 9 complex and 14 real sizes from 20480 to 663552 where `pffft_zconvolve`
measured 0.95-0.97x in at least one sweep, although the transforms there
were 1.23-1.65x as fast. The kernel is the 4-lane loop on 256-bit vectors
and memory bound at these sizes; the conservative rule keeps the 4-lane
code rather than trade a few percent in one operation.

The fused layout passes remove one sweep over the data from the 4-lane
complex transforms (real transforms have no such pass and are unchanged,
0.96-1.04x). Out of place, float complex transforms became 1.03-1.18x as
fast at every measured N from 96 to 2^20 (N = 64: 0.97-1.15x), double
1.00-1.15x from N = 128 with one exception, ordered backward at N = 1024
(0.95x). In place, the gain depended on the parity of the pass count: at
the sizes where the fused first pass would have had to write into its own
input (float N = 384..1280, 6144..16384, 2^17..2^18; double N = 1024,
16384) the transform fell back to the separate sweep and stayed at
0.97-1.02x. For unordered transforms the in-place change below removes
that fallback; ordered in-place transforms keep it, since their combined
finalize-and-reorder step cannot run in place. At N = 64 double complex
measured 0.91-0.95x in the runner-built drivers but equal in a separately
linked binary of the same code, where disabling the fusion was slower:
code-layout sensitivity at a 5 ns scale, not a cost of the fusion.

The tables were measured with the passes before the [terminal radix-8 pass
and the in-place first pass](#terminal-radix-8-pass-and-in-place-first-pass)
and were not measured again for them, so every size keeps the lane count it
had ([Size tables](#size-tables)).

Against other libraries see [Other FFT libraries](#other-fft-libraries).

### Other FFT libraries

`bench_pffft_float` (MFlops; median of three runs, run-to-run spread mostly
under 10%) of the final code as packages build it (`TARGET_C_ARCH=none`, the
8-lane code at the sizes of the Zen 3 desktop table at run time) on the
5950X, CPU 14, against FFTW 3.3.10 built with `--enable-sse2 --enable-avx
--enable-avx2` (float, `FFTW_MEASURE` and `FFTW_ESTIMATE` plans) and the
bundled Green, Kiss and Pocket FFTs (same baseline flags);
[tables](results/library-comparison.txt). Ratios of PFFFT's unordered
transform (PFFFT-U; ordered output costs up to 16% more) to the others:

| | FFTW MEASURE | FFTW ESTIMATE | Pocket | Green | Kiss |
| --- | --- | --- | --- | --- | --- |
| complex, N = 2^k | 0.70-0.91x at 32..128 and 4096; 0.97-1.15x at 256..2048; 1.19-1.56x at 16 and 8192..262144 | 1.19-4.89x | 2.3-5.0x | 3.8-9.3x | 4.0-18x |
| real, N = 2^k | 0.92x at 256; 1.28-1.79x at every other N from 32 to 262144 | 1.47-3.17x | 2.0-6.9x | 3.0-11.7x | 3.4-18x |
| complex, other N | 0.54-0.79x (96..384, 640, 768, 2400, 9216); 1.04-1.30x (480, 800) | 1.11-3.16x | 3.6-4.8x | 6.8-9.3x | 11-18x |
| real, other N | 0.62-0.74x (160, 192, 9216); 0.99-1.41x (96, 384..2400) | 1.15-3.51x | 2.8-6.1x | 5.2-9.7x | 8.0-18x |

So FFTW with measured plans remains faster for short and for most
non-power-of-two complex transforms, where its generated codelets and
planner choose better radix sequences than PFFFT's fixed FFTPACK passes (a
host-tuned PFFFT build with AVX2 and FMA throughout is within 0.05x of these
ratios for complex transforms: the gap is not the instruction set); PFFFT
leads for real transforms and large power-of-two complex ones, and is
faster than FFTW with estimated plans everywhere. With the table of the
unmeasured Zen 3 models (the sizes of both measured ones) complex
N = 65536 and 262144 and real N = 262144 measured 0.96-0.97x instead of
1.46-1.56x, complex N = 64 0.60x instead of 0.74x. Small sizes depend on
code layout: between two builds that differ only in the generated size
tables, real N = 96 and complex N = 16, which use the 4-lane code in both,
measured 22300 and 30100 MFlops and 25600 and 22300 MFlops, reproducibly;
take single small-size ratios as uncertain by that much. Not compared: FFTW
with AVX-512 (Zen 3 has none), Intel MKL, Apple vDSP, double precision,
other CPUs.

### In-place unordered complex transforms

With an even number of FFTPACK passes, an in-place unordered complex
transform used to hit the parity at which the fused first (forward) or last
(backward) pass would have had to write into its own input, and fell back
to a separate deinterleave or interleave sweep. `pffft_transform()` now
sends exactly these calls (4-lane, complex, input == output, even pass
count) to a separate function: forward, the first pass reads the data and
writes the scratch buffer and the unordered finalize ends in the data
buffer, in place if the passes end there; backward, the unordered
preprocess runs in place. Both work block by block, reading a block's
vectors before writing them. Every other call runs the unchanged driver,
whose machine code is identical to before (float: up to alignment padding;
double: byte for byte); only the small entry function, into which the
compiler inlines the driver, differs in register allocation.

A/B against the same tree without the change (host-tuned 4-lane builds, 5
interleaved samples per case, at most 2.5 busy CPUs;
[samples](results/inplace-parity-ab.csv.gz)), unordered in place, at the
sizes with an even pass count (N = 64, 384..1280, 6144..16384):

- float: 1.01-1.13x (1.04-1.13x from N = 384); in place now takes 0.94-1.03x
  the time of out of place (was 1.01-1.13x);
- double: 1.05-1.18x at N = 64, 384..640 and 6144..16384; 0.91-1.04x at
  768..1280.

Timings at double N = 768..1280 depend on where the process's pages land:
each process is stable, but processes differ by up to 37% in the new code
(up to 16% before), also with address randomization disabled, so presumably
physical placement. Over 30 processes per build
([samples](results/inplace-parity-processes.csv.gz)) the mean ratio is
0.99-1.01x at 768 and 1024 and 0.94x (forward) and 0.97x (backward) at
1280, with the slowest decile at 0.88-0.89x; the change costs there. The
same measurement at other even-pass sizes, including ones not in the A/B,
gives double 1.05-1.08x (N = 384, 640, 960), 1.11-1.15x (1600, 4320, 8192)
and float 1.06-1.09x (960..4320), so the change is kept for all sizes; no
simple rule (buffer size, radix) separates the three slower double sizes.

Paths the change does not alter measured 0.93-1.03x (float) and 0.91-1.23x
(double); the deviations repeat within a build but not across builds. Float
N = 96 and 128 forward, in and out of place, are 0.92-0.97x with
non-overlapping interquartile ranges (15 samples,
[confirmation](results/inplace-parity-confirm.csv.gz)), although their
driver code is unchanged. We attribute this to code layout, an inference
from the following: the first version of this change, which modified the
shared driver, showed the same deficit in the backward direction at these
sizes instead ([samples](results/inplace-parity-ab-v1.csv.gz),
[confirmation](results/inplace-parity-confirm-v1.csv.gz)), and rebuilding
the unchanged tree with `-falign-functions=64` alone moved its N = 96
backward time from 75 to 91 ns
([control](results/inplace-parity-align-control-v1.csv.gz)). On Zen 3,
float N = 128 uses the 8-lane code anyway.

### CPU classes

`pffft_x86_wide_class()` (src/pffft_cpu_x86.h) sorts AVX2 CPUs by CPUID
vendor, family and model into explicit lists, with family/model ranges from
LLVM's `Host.cpp` (cross-checked with AMD's family/model table and Linux
`intel-family.h`): *measured*, each with its own size table (Zen 3 desktop
and server models and the unmeasured Zen 3 APUs: the tables above; Zen 4:
see below), *inferred* (full-width 256-bit FP datapaths,
no AVX frequency licences: Zen 2, Zen 3 Badami, Zen 5, Intel client big
cores Haswell..Rocket Lake and P-core-only Alder/Raptor Lake-S) and *off*
(everything else, including 128-bit datapaths, hybrids, Intel servers,
unknown models and future families). By default only the measured class uses
the 8-lane code; the inferred class is opt-in (`PFFFT_WIDE_INFERRED`) and
then uses a stricter table (every gated mode at least 1.25x in every
measured sweep, Zen 3 and Zen 4: 81 complex, 67 real sizes) and only sizes
whose working set (input, output, work and twiddles: 32N bytes complex, 16N
real) fits in L2, read from CPUID leaf 4 (Intel) or 8000_001Dh/8000_0006h
(AMD). Unit tests cover every range boundary and hole; Intel SDE runs in CI
check the classes against the CPUIDs of 22 emulated Intel chips
(`tests/run_sde_checks.sh`).

The inferred class was meant to be backed by a static throughput model
([mca_predict.py](mca_predict.py)): predicted cycles per call = sum over the
basic blocks of (executions per call, counted with valgrind's callgrind on
the shipping baseline GCC build) x (llvm-mca 23.1.3 steady-state cycles of
the block on the CPU's scheduling model), for the 56 L1-resident cases of
the inferred table, all eight modes. Its rule was fixed before the data:
calibration within 15% of the speedups measured on Zen 3 in every case, a
Zen 1 control (256-bit operations split) at most 1.05x after correction for
the model's bias, and then every inferred model at least 1.10x. Result
([summary](results/mca-crosscheck.txt), [cases](results/mca-crosscheck.csv.gz)):
calibration deviation median 4.8% but up to 18.9% (fail), Zen 1 predicted
1.42x (fail: the model cannot tell the datapath widths apart; the gain it
predicts comes from fewer instructions, loads and stores per element). The
other models predict 1.35-1.77x at their worst case, but under the rule
that is no evidence, so the inferred class stays off by default.

### Measuring another CPU

A CPU model joins the measured class with two sweeps like the ones above on
it: a `native4` build (`-DPFFFT_ENABLE_WIDE_DISPATCH=OFF`) and a `wide` build
(`-DPFFFT_WIDE_ALL_SIZES=ON`: the 8-lane code at every size it supports),
`run_matrix.py --interleave --repeats 3` over all supported sizes and the
eleven operations, each sweep with a `--seed` of its own so that the sweeps'
orders are independent, then `wide_size_table.py` and a per-model entry in
`pffft_x86_wide_class()`. The workflow
[bench-runner-sweep.yml](../../.github/workflows/bench-runner-sweep.yml)
runs such sweeps on GitHub-hosted runners (virtual machines whose CPU varies
between AMD and Intel models) and records the CPU of each.

Two batches of six such sweeps (host-tuned GCC builds, 4 vCPUs, host-load
gate inside the VM; [CPUs](results/runner-cpus.txt)) landed on four CPUs.
Each figure below is the ratio of the medians (`native4` time / `wide`
time) of one case (transform type, N, operation) in one sweep, over all
eleven operations and every sweep of the CPU:

| CPU | sweeps | passes the rule in its own sweeps (complex / real sizes) | at the sizes it now uses |
| --- | ---: | ---: | --- |
| Zen 4, EPYC 9V74 (19h model 11h) | 3 ([1](results/runner-zen4-epyc9v74-1.csv.gz), [2](results/runner-zen4-epyc9v74-2.csv.gz), [3](results/runner-zen4-epyc9v74-3.csv.gz)) | 186 / 163 | Zen 4 table: median 1.38x, lowest 0.97x |
| Zen 3, EPYC 7763 (19h model 01h) | 7 ([1](results/runner-zen3-epyc7763-1.csv.gz), [2](results/runner-zen3-epyc7763-2.csv.gz), [3](results/runner-zen3-epyc7763-3.csv.gz), [4](results/runner-zen3-epyc7763-4.csv.gz), [5](results/runner-zen3-epyc7763-5.csv.gz), [6](results/runner-zen3-epyc7763-6.csv.gz), [7](results/runner-zen3-epyc7763-7.csv.gz)) | 143 / 115 | Zen 3 server table: median 1.44x, lowest 0.97x |
| Zen 5, EPYC 9V45 (1Ah model 02h) | 1 ([1](results/runner-zen5-epyc9v45-1.csv.gz)) | (one sweep) | none by default; opted in, the inferred table within its 1 MB L2: median 1.53x, lowest 1.07x |
| Intel Xeon Platinum 8573C | 1 ([1](results/runner-intel-xeon8573c-1.csv.gz)) | (one sweep) | none (Intel server parts are off) |

Zen 4 is therefore a measured class with its own table from its three
sweeps (`pffft_wide_zen4_*` in `pffft_wide_sizes.h`). The Zen 3 server part
does not confirm the desktop table: at its sizes real `pffft_zreorder` at
N = 128 measured 0.82x in the first three sweeps and complex
`pffft_zconvolve` at N = 288000 0.78x in one. So the server model has its
own table from its seven sweeps, the desktop model keeps its own, the
unmeasured Zen 3 models use the sizes of all nine Zen 3 sweeps (a size must
pass in each), and the inferred table comes from every sweep of every
measured group (1.25x). The second batch was
first held out; checked against the tables of the first, it found
enabled sizes that fail the rule (Zen 3: real `pffft_zconvolve` 0.96x at
N = 240000; Zen 4: complex `pffft_zconvolve` 0.93x at N = 24000, with
disjoint interquartile ranges), so under the rule it is part of the tables
now, which removed 1 complex and 8 real sizes from the Zen 3 server table
(144 / 123 from the first three sweeps) and 3 complex and 1 real from the
Zen 4 table. Zen 5 stays inferred: one sweep is not enough for a
table, and at the sizes of the measured tables it has cases below 1.0x
(Zen 3 table: 21, down to 0.90x; Zen 4 table: 32, down to 0.88x, mostly
large-N pipeline and spectral multiplication), so those tables would be
wrong for it; the inferred table within its L2, which this sweep did not
take part in, is never slower there. On the Xeon the measured tables would
be wrong too (real `pffft_zreorder` 0.81-0.83x at N = 384..1024), which
supports keeping Intel server parts off. These are virtual machines on
shared hosts: evidence for the tested models, an extrapolation for other
models of the same microarchitecture. More sweeps are not run in the hope
of landing on other CPUs: the runners' CPU cannot be chosen. The tables are
generated with
`wide_size_table.py --output src/pffft_wide_sizes.h --group zen3 DESKTOP... SERVER... --group zen3_desktop DESKTOP... --group zen3_server SERVER... --group zen4 ...`.

### Windows

GCC and Clang build the 8-lane code with target pragmas on Windows as
elsewhere (MinGW-w64 GCC 13 and 15, clang-cl 19 in CI). The pragmas are not
what makes it safe: every vector load and store is an explicit unaligned
`VLD`/`VST` (the 4-lane, scalar, NEON, Altivec and WASM objects were shown
unchanged by comparing disassembly with relocations and the referenced
constant data before and after), and `check_isa_baseline.py` proves the
alignment of every remaining aligned access (stack slots tracked modulo 32,
RIP-relative constants by section alignment and relocation) on ELF and
PE/COFF objects alike.

MSVC was first built with the 8-lane code inside `pffft.c` and its AVX2
intrinsics compiled without `/arch` (same translation unit). That works
(all tests, ISA and SSE/AVX transition audit clean in optimized builds), but
the generated code is slower. Interleaved measurement on the Zen 3 host
(CPU 14, MSVC 19.44 binaries under wine, 5 samples per case and build,
host-load gate; [samples](results/windows-gate.csv.gz),
[table](results/windows-gate.txt)), speedup over the 4-lane code of the same
compiler at the 144 table-size cases (9 sizes, all 9 gated modes):

| Build | complex min / median | real min / median |
| --- | ---: | ---: |
| GCC baseline | 1.30 / 1.65x | 1.17 / 1.71x |
| MSVC, same translation unit, no `/arch` | 1.23 / 1.42x | 1.12 / 1.31x |
| MSVC, whole library with `/arch:AVX2` (experiment) | 1.28 / 1.59x | 1.25 / 1.55x |

The agreed criterion for the same-translation-unit approach was a speedup
within 10% of GCC's in every case; it was within 10% in 26 of 144 (median
0.84 of GCC's), while the whole-library `/arch:AVX2` experiment was within
10% in 111 of 144. MSVC therefore builds the 8-lane code as its own
translation unit, `pffft_wide_tu.c`, with `/arch:AVX2`, and `pffft.c` for the
baseline. Measured the same way as that build ([samples](results/windows-gate-tu.csv.gz),
[table](results/windows-gate-tu.txt), 4-lane code of the baseline MSVC build
as the reference): complex 1.28 / 1.59x, real 1.25 / 1.56x (min / median),
within 10% of GCC's speedup in 114 of 144 cases (median 0.96), never below
1.25x. Rules for that file: it holds nothing that
runs before the CPU check (the audit rejects any non-8-lane function with
AVX code in it, which caught MSVC Debug emitting `cosl`/`sinl` copies that
the linker could have shared with `pffft.c`); it is compiled without `/GL`;
and the CPU check also requires the rest of x86-64-v3.

### Tried and rejected

| Candidate | Result |
| --- | --- |
| Radix-8 first pass replacing the radix-2 and one radix-4 pass (8-lane complex, N/8 = 2·4^k) | 0.80-1.19x, no consistent gain ([samples](results/wide-radix8-rejected.csv.gz)); consistent with the arm64 radix-8 result in 2ce8a89. That experiment replaced the first pass; the [terminal radix-8 pass](#terminal-radix-8-pass-and-in-place-first-pass) replaces the last, where it needs no twiddles |
| 8x8 transpose with the 128-bit exchange done by `vinsertf128` loads | 0.96-1.04x: finalize is not bound by the shuffle ports alone |
| Half-length complex FFT for real input on 8 lanes | not built: same pass work as the 8-way real combination plus two extra sweeps |
| Split-radix / conjugate-pair, four-step / cache-oblivious, planner | not attempted: no measured speed gain in the literature at these sizes, PFFFT already leads FFTW at large N |

### Correctness and portability evidence

- `test_pffft_accuracy_float` runs every size on both backends against long
  double reference DFTs (through N = 65536: 8-lane worst errors 0.11 eps*log2N
  forward and backward), including bitwise ordered/unordered equivalence,
  in-place and `work == NULL`.
- `test_pffft_wide`: CPU detection truth table and probe order with fake
  CPUID/XGETBV, setup selection, every operation with input, output and work
  at 16 mod 32 bytes against aligned results and guard regions, and
  concurrent setup creation.
- `check_isa_baseline.py` (a ctest in baseline x86-64 Linux and Windows
  builds, GNU objdump on ELF and PE/COFF): outside the 8-lane functions only
  x86-64 baseline instructions (VEX, EVEX and XOP told by their encoding, so
  VEX-encoded BMI is caught, and SSE3..SSE4.2, POPCNT, LZCNT, MOVBE and
  others by mnemonic); inside them only what the CPU check checks; every
  instruction that needs an aligned operand provably aligned; and no
  legacy-SSE instruction, call out of or return from the 8-lane code while
  the upper ymm halves may be dirty (data flow through calls between 8-lane
  functions; indirect jumps only as jump tables shown local through ELF or
  COFF relocations). COFF objects are read from their own section, symbol
  and relocation tables, since GNU objdump mislabels MSVC's same-named
  COMDAT sections; linked MSVC images (also under LTCG) through the linker
  map. 28 assembly fixtures are its negative and positive controls.
  It also found that the previous version passed vacuously with
  llvm-objdump (which CMake picks for Clang builds); it now requires GNU
  objdump and fails on output without instructions.
- Intel SDE 10.13.1 in CI (Linux and MSVC/clang-cl builds, static and DLL):
  the tests pass on emulated Nehalem and Sandy Bridge without executing an
  instruction those chips lack, and an unguarded AVX2 instruction in an
  executable and in a DLL is reported; on emulated Haswell, SDE's AVX/SSE
  transition checker counts no transition while test_pffft_wide runs the
  8-lane code (0 in 17.5 billion instructions for the GCC baseline build;
  1000 of 1000 in its negative control).
- `tests/run_emulated_cpu_tests.sh` (CI job `x86_cpu_fallback`): static
  baseline binaries pass under QEMU emulating Nehalem and SandyBridge without
  selecting the backend (QEMU raises SIGILL for AVX2 there), and use and pass
  it on Haswell.

### Limitations

- Measured on Zen 3 (a desktop part, confirmed on a server part in VMs) and
  Zen 4 (server part, in VMs only); other AVX2 CPUs keep the 4-lane code
  unless built with `PFFFT_WIDE_INFERRED=ON` (see CPU classes) or
  `PFFFT_WIDE_UNKNOWN_CPUS=ON`. The Zen 4 table comes from GitHub-hosted
  virtual machines, whose neighbours and clocks are not controlled.
- With MSVC, projects that do not use this CMake build must add
  `pffft_wide_tu.c` (compiled with `/arch:AVX2`) and define `PFFFT_WIDE_TU`;
  otherwise MSVC builds keep the 4-lane code. The MSVC measurement ran under
  wine on Zen 3, not on Windows.
- SSE/AVX transitions are checked statically (skipped for unoptimized builds)
  and with SDE's emulated transition checker; no Intel hardware was available
  to measure their cost.
- Double precision stays on its existing 4-lane AVX code; AVX-512 is not used.
- Buffers at 16 mod 32 bytes are correct but lose part of the gain
  (prototype: 1.15-1.67x instead of 1.4-1.9x).

## Terminal radix-8 pass and in-place first pass

Two changes to the pass schedule of
[pffft_passes_impl.h](../../src/pffft_passes_impl.h), each a compile-time
switch of [pffft_pass_config.h](../../src/pffft_pass_config.h):

- **Terminal radix-8 pass** (`PFFFT_TERMINAL_RADIX8`). A plan has no
  stand-alone radix-2 pass any more: where it had a radix-2 and a radix-4
  pass, the radix-2 pass is dropped and the last radix-4 pass becomes one
  twiddle-free radix-8 pass (complex factors 2, ..., 4 become ..., 8; real
  power-of-two factors 2, 4, ..., 4 become 4, ..., 8). The passes in between
  run with other `l1` and `ido`, and the transform makes one sweep over the
  data fewer. Radix 8 is used only as the pass with one element per leg: the
  last pass of a complex transform, the first pass of the real forward
  transform and the last of the real backward one. Only there does it need no
  twiddles; as a twiddled pass in an earlier position it measured slower
  (the radix-8 first pass under [Tried and rejected](#tried-and-rejected),
  and the arm64 measurements described at `cffti1_ps`). Real sizes with a
  factor 3 or 5 keep their radix-2 pass: FFTPACK's odd-radix real passes
  need an odd `ido`, so those factors come last in a real plan, and a
  radix-8 pass after them would need twiddles.
- **In-place first pass** (`PFFFT_INPLACE_FIRST_PASS`). Complex transforms
  on the backends with the fused layout passes (every 4-lane backend and the
  8-lane AVX2 one): where the parity of the pass count would otherwise cost
  an in-place transform an extra sweep to deinterleave or interleave its
  data, the first pass runs in place. In the first pass every butterfly
  reads and writes the same positions, so it can; running it in place
  exchanges the buffers of the later passes, and the separate sweep is not
  needed. Real transforms are unchanged.

Both are on by default on every target and backend, with one exception. A
change that does less work per transform applies everywhere unless a
measurement speaks against it on a backend, and on the single-lane scalar
build (`-DPFFFT_USE_SIMD=OFF -DPFFFT_USE_SCALAR_VECT=OFF`, backend `Scalar`,
`SIMD_SZ == 1`) one does: with the terminal radix-8 pass complex transforms
measured slower there, while real transforms gained
([below](#slower-cases)). The switch is one per build, so on that build the
terminal radix-8 pass is off by default; the in-place first pass compiles to
nothing there, so that build runs the passes it ran before. The 4xScalar
build, which CMake builds by default with `PFFFT_USE_SIMD=OFF`, keeps both.
CMake: `-DPFFFT_TERMINAL_RADIX8=AUTO|ON|OFF` and
`-DPFFFT_INPLACE_FIRST_PASS=AUTO|ON|OFF` (`AUTO`, the default, leaves the
choice to `pffft_pass_config.h`; a parent project's normal variable of the
same name is honoured). Other build systems may define either macro to 0
or 1. With both off the passes are those before this change.

At the sizes whose plan changes, results differ in the last bits from those
of an earlier build: the accuracy tests pass, a bit-exact comparison
against an earlier build would not. `test_pffft_passes_float` and
`test_pffft_passes_double` ([test_pffft_passes.c](../../tests/test_pffft_passes.c))
test the codelets, their drivers, the plans and the defaults of the switches
directly.

### Survey

There is no exhaustive per-size acceptance run behind this change; it was
merged on a survey. Old (master `c3b3005`) against new, both libraries
built on the machine that measures, the same driver (`bench_hotloops.cpp`)
linked against each, `run_matrix.py --interleave --repeats 5 --seconds 0.02`:
one fresh process per sample, the two builds of a repeat back to back in
random order. 40 complex sizes from 64 to 1,048,576 and 35 real sizes from
128 to 1,048,576, all eleven operations, float and double. A case is a
transform type, N and operation; its figure is the median of its five pair
ratios, old time / new time, so above 1 the new code is faster. *Changed
cases* are the sizes and operations whose pass plan or buffer routing
changes; *other cases* the rest (unchanged plan, `pffft_zreorder`,
`pffft_zconvolve`). Seven CPU models were timed; a Ryzen 9 5950X ran only
the code-placement diagnostic (item 1 under [Slower cases](#slower-cases)).

The survey ran while the change was being finished, so its new libraries
are development revisions whose pass code is that of the merge
(`pffft_passes_impl.h` differs from the merged file in one comment). On the
x86 SIMD backends both switches were already on by default in that
revision; on every other backend (NEON on the Neoverse-N2 and the
Cortex-A53, 4xScalar and single-lane scalar on the Xeon, single-lane scalar
on the Neoverse-N2) it still had them off by default, and the new library
was built with `-DPFFFT_TERMINAL_RADIX8=ON -DPFFFT_INPLACE_FIRST_PASS=ON`,
which for NEON and 4xScalar compiles what the merged default compiles. The
default build as shipped on the Ryzen 7 5800HS was built from the merge
candidate itself, which differs from the merged code in the single-lane
scalar default only and has `master`'s size tables. The development
revision has other 8-lane size tables than `master`, a regenerated set that
is not part of this change. That matters only where both builds dispatch on
a CPU that has a table: on the EPYC 7763 and 9V74 some float sizes ran 4
lanes in one build and 8 lanes in the other, and those cases are left out
of the table (item 6). Everywhere else the two builds ran the same lane
count.

| CPU | compiler, conditions | builds (raw samples) |
| --- | --- | --- |
| Ryzen 7 5800HS (Zen 3, Cezanne) | GCC 16.2.1, CPU 14 with its SMT sibling idle, host load at most 2.5 CPUs | host-tuned, 4 lanes, no dispatch ([samples](results/pass-schedule-survey-zen3-5800hs-w4.csv.gz)); host-tuned, 8 lanes at every size ([samples](results/pass-schedule-survey-zen3-5800hs-w8.csv.gz)); host-tuned default as shipped, dispatch with the common Zen 3 size table, float only ([samples](results/pass-schedule-survey-zen3-5800hs-disp.csv.gz)) |
| Xeon E5-2697 v3 (Haswell-EP) | GCC 16.2.1, one core per socket, SMT sibling idle | host-tuned default ([samples](results/pass-schedule-survey-haswell-e5-2697v3-native.csv.gz)); x86-64 baseline ([samples](results/pass-schedule-survey-haswell-e5-2697v3-x8664.csv.gz)); 4xScalar ([samples](results/pass-schedule-survey-haswell-e5-2697v3-scalar4.csv.gz)); single-lane scalar ([samples](results/pass-schedule-survey-haswell-e5-2697v3-scalar.csv.gz)); host-tuned, 8 lanes at every size, float only, not what this CPU runs by default: no size table covers it ([samples](results/pass-schedule-survey-haswell-e5-2697v3-wide8.csv.gz)) |
| Cortex-A53 (Raspberry Pi 3 Model B) | GCC 14.2, CPU 3, `ondemand` governor | default, NEON; float complex, float real, double complex ([samples](results/pass-schedule-survey-cortex-a53-rpi3-neon.csv.gz)) |
| Neoverse-N2 (three GitHub-hosted ARM64 runners, 4 vCPUs) | GCC 13.3.0, CPU 1, host load at most 2.5 CPUs | default, NEON ([1](results/pass-schedule-survey-neoverse-n2-runner1-neon.csv.gz), [2](results/pass-schedule-survey-neoverse-n2-runner2-neon.csv.gz), [3](results/pass-schedule-survey-neoverse-n2-runner3-neon.csv.gz)); single-lane scalar ([1](results/pass-schedule-survey-neoverse-n2-runner1-scalar.csv.gz), [2](results/pass-schedule-survey-neoverse-n2-runner2-scalar.csv.gz), [3](results/pass-schedule-survey-neoverse-n2-runner3-scalar.csv.gz)) |
| EPYC 7763 (Zen 3 server; three GitHub-hosted runners, 4 vCPUs) | GCC 13.3.0, CPU 1, host load at most 2.5 CPUs | host-tuned default ([1](results/pass-schedule-survey-epyc7763-runner1-native.csv.gz), [2](results/pass-schedule-survey-epyc7763-runner2-native.csv.gz), [3](results/pass-schedule-survey-epyc7763-runner3-native.csv.gz)); x86-64 baseline ([1](results/pass-schedule-survey-epyc7763-runner1-x8664.csv.gz), [2](results/pass-schedule-survey-epyc7763-runner2-x8664.csv.gz), [3](results/pass-schedule-survey-epyc7763-runner3-x8664.csv.gz)) |
| EPYC 9V74 (Zen 4 server; one GitHub-hosted runner, 4 vCPUs) | GCC 13.3.0, CPU 1, host load at most 2.5 CPUs | host-tuned default ([samples](results/pass-schedule-survey-epyc9v74-runner1-native.csv.gz)); x86-64 baseline ([samples](results/pass-schedule-survey-epyc9v74-runner1-x8664.csv.gz)) |
| EPYC 9V45 (Zen 5 server; two GitHub-hosted runners, 4 vCPUs) | GCC 13.3.0, CPU 1, host load at most 2.5 CPUs | host-tuned default ([1](results/pass-schedule-survey-epyc9v45-runner1-native.csv.gz), [2](results/pass-schedule-survey-epyc9v45-runner2-native.csv.gz)); x86-64 baseline ([1](results/pass-schedule-survey-epyc9v45-runner1-x8664.csv.gz), [2](results/pass-schedule-survey-epyc9v45-runner2-x8664.csv.gz)) |

The single-lane scalar builds were measured with the terminal radix-8 pass
forced on, the measurement that decided its default there. The [invocation records](results/pass-schedule-survey-records.txt)
give each file's start and end, rows, seed and the SHA-256 of both driver
binaries. To repeat one part of the survey (one transform type and
precision per invocation, each with a seed of its own):

```sh
mkdir -p build_old_src
git archive c3b3005 | tar -x -C build_old_src
cmake -S build_old_src -B build_old -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build_new -DCMAKE_BUILD_TYPE=Release
cmake --build build_old --parallel
cmake --build build_new --parallel
SIZES=$(zcat benchmarks/perf_analysis/results/pass-schedule-survey-haswell-e5-2697v3-native.csv.gz |
        awk -F, '$4 == "complex" {print $6}' | sort -nu)
python3 benchmarks/perf_analysis/run_matrix.py \
  --build base=build_old --build new=build_new \
  --output pass-schedule-complex-float.csv --cpu CPU --idle-sibling \
  --max-busy-cpus 2.5 --interleave --repeats 5 --seconds 0.02 --seed SEED \
  --kinds complex --precisions float --sizes $SIZES \
  --ops forward backward ordered-forward ordered-backward inplace-forward \
        inplace-backward inplace-ordered-forward inplace-ordered-backward \
        reorder zconv pipeline
```

This builds the merged tree, where the host-tuned default builds need no
option. For the other builds, add to both configurations:
`-DPFFFT_ENABLE_WIDE_DISPATCH=OFF` (4 lanes, no dispatch),
`-DPFFFT_WIDE_ALL_SIZES=ON` (8 lanes at every size),
`-DTARGET_C_ARCH=x86-64 -DTARGET_CXX_ARCH=x86-64` (x86-64 baseline),
`-DPFFFT_USE_SIMD=OFF` (4xScalar), `-DPFFFT_USE_SIMD=OFF
-DPFFFT_USE_SCALAR_VECT=OFF` (single-lane scalar; `-DPFFFT_TERMINAL_RADIX8=ON`
for the new build only, since `c3b3005` has no such option).

Each cell gives, over the changed cases of a row, the geometric mean of the
ratios, the lowest and highest single case, and how many cases are below
1.00. The last column gives the geometric mean over the other cases,
complex / real, with the number of other cases below 0.97 in brackets.
Backend labels are those the driver prints (`pffft_simd_arch()` and the lane
count). Where several runners of one CPU model (Neoverse-N2, EPYC 7763,
EPYC 9V45) agree, the row gives the range of their figures. On the EPYC
7763 and 9V74 the float cells also give the number of cases left out
(item 6); they are neither changed nor other cases. "—": not measured.

| CPU | build | precision, backend | complex, changed cases | real, changed cases | other cases, complex / real |
| --- | --- | --- | --- | --- | --- |
| Ryzen 7 5800HS (Zen 3) | host-tuned, 4 lanes, no dispatch | float, SSE1-x4 | 1.071 (0.93-1.23; 13 of 191) | 1.084 (0.96-1.27; 10 of 63) | 1.001 (7) / 1.000 (4) |
| Ryzen 7 5800HS (Zen 3) | host-tuned, 4 lanes, no dispatch | double, AVX-x4 | 1.090 (0.93-1.40; 7 of 191) | 1.135 (0.98-1.35; 3 of 63) | 0.978 (44) / 1.002 (3) |
| Ryzen 7 5800HS (Zen 3) | host-tuned, 8 lanes at every size | float, SSE1-x8 (complex also SSE1-x4) | 1.104 (0.81-1.40; 2 of 238) | 1.118 (0.98-1.33; 4 of 63) | 0.995 (10) / 0.999 (11) |
| Ryzen 7 5800HS (Zen 3) | host-tuned default as shipped (dispatch, common Zen 3 size table) | float, SSE1-x4 and SSE1-x8 | 1.106 (0.82-1.38; 3 of 206) | 1.122 (0.98-1.32; 2 of 45) | 0.997 (13) / 1.000 (5) |
| Xeon E5-2697 v3 (Haswell-EP) | host-tuned default | float, SSE1-x4 | 1.136 (1.01-1.27; 0 of 191) | 1.148 (0.96-1.35; 1 of 63) | 1.000 (2) / 1.000 (9) |
| Xeon E5-2697 v3 (Haswell-EP) | host-tuned default | double, AVX-x4 | 1.146 (0.97-1.28; 1 of 191) | 1.149 (1.03-1.35; 0 of 63) | 1.000 (7) / 1.001 (11) |
| Xeon E5-2697 v3 (Haswell-EP) | x86-64 baseline | float, SSE1-x4 | 1.116 (1.01-1.23; 0 of 191) | 1.124 (0.96-1.34; 1 of 63) | 1.000 (4) / 0.999 (9) |
| Xeon E5-2697 v3 (Haswell-EP) | x86-64 baseline | double, SSE2-x4 | 1.106 (0.99-1.28; 1 of 191) | 1.109 (0.98-1.31; 2 of 63) | 1.000 (3) / 0.999 (6) |
| Xeon E5-2697 v3 (Haswell-EP) | 4xScalar (the default without SIMD) | float, 4xScalar-x4 | 1.070 (0.95-1.24; 14 of 191) | 1.152 (0.89-1.42; 3 of 63) | 0.999 (3) / 1.000 (3) |
| Xeon E5-2697 v3 (Haswell-EP) | 4xScalar (the default without SIMD) | double, 4xScalar-x4 | 1.059 (0.89-1.26; 37 of 191) | 1.131 (0.83-1.41; 1 of 63) | 1.000 (3) / 1.001 (8) |
| Xeon E5-2697 v3 (Haswell-EP) | single-lane scalar, terminal radix-8 pass forced on (off by default) | float, Scalar-x1 | 0.942 (0.84-1.08; 144 of 171) | 1.152 (1.04-1.40; 0 of 63) | 0.996 (13) / 0.999 (2) |
| Xeon E5-2697 v3 (Haswell-EP) | single-lane scalar, terminal radix-8 pass forced on (off by default) | double, Scalar-x1 | 0.994 (0.85-1.16; 93 of 171) | 1.130 (1.04-1.25; 0 of 63) | 1.002 (1) / 1.000 (3) |
| Xeon E5-2697 v3 (Haswell-EP) | host-tuned, 8 lanes at every size (no size table covers this CPU) | float, SSE1-x8 (complex also SSE1-x4) | 1.151 (1.05-1.38; 0 of 238) | 1.133 (1.03-1.29; 0 of 63) | 1.003 (3) / 1.003 (6) |
| Cortex-A53 (Raspberry Pi 3) | default (NEON) | float, NEON-x4 | 1.056 (0.82-1.26; 38 of 191) | 1.100 (0.92-1.28; 4 of 63) | 0.998 (5) / 0.997 (12) |
| Cortex-A53 (Raspberry Pi 3) | default (NEON) | double, NEON-x4 | 1.052 (0.87-1.25; 22 of 191) | — | 1.000 (4) / — |
| Neoverse-N2 (3 runners) | default (NEON) | float, NEON-x4 | 1.078 (1.03-1.19; 0 of 191) | 1.068-1.069 (0.98-1.16; 3-4 of 63) | 0.999-1.000 (0-1) / 0.998 (0-4) |
| Neoverse-N2 (3 runners) | default (NEON) | double, NEON-x4 | 1.076-1.077 (1.00-1.19; 0 of 191) | 1.058 (0.85-1.27; 10 of 63) | 0.999-1.001 (3-6) / 1.000-1.002 (0-4) |
| Neoverse-N2 (3 runners) | single-lane scalar, terminal radix-8 pass forced on (off by default) | float, Scalar-x1 | 0.978 (0.93-1.01; 142 of 171) | 1.081 (1.00-1.30; 0 of 63) | 0.999-1.000 (2-5) / 1.007-1.008 (0-3) |
| EPYC 7763 (Zen 3 server, 3 runners) | host-tuned default | float, SSE1-x4 and SSE1-x8 | 1.134-1.137 (1.00-1.43; 0-1 of 193; 55 left out) | 1.132-1.137 (0.98-1.34; 1-4 of 45; 22 left out) | 0.992-0.996 (15-19) / 0.997 (9-12) |
| EPYC 7763 (Zen 3 server, 3 runners) | host-tuned default | double, AVX-x4 | 1.107-1.117 (0.96-1.40; 2 of 191) | 1.162-1.163 (0.91-1.37; 2 of 63) | 1.001-1.002 (8-16) / 1.001-1.003 (4-8) |
| EPYC 7763 (Zen 3 server, 3 runners) | x86-64 baseline | float, SSE1-x4 and SSE1-x8 | 1.134-1.137 (1.01-1.38; 0 of 193; 55 left out) | 1.130-1.131 (0.97-1.34; 0-2 of 45; 22 left out) | 0.995-0.997 (7-10) / 0.998-0.999 (9-11) |
| EPYC 7763 (Zen 3 server, 3 runners) | x86-64 baseline | double, SSE2-x4 | 1.078-1.083 (0.98-1.23; 0-1 of 191) | 1.103-1.118 (0.97-1.39; 2-3 of 63) | 0.999-1.002 (6-9) / 1.001-1.002 (5-8) |
| EPYC 9V74 (Zen 4 server) | host-tuned default | float, SSE1-x4 and SSE1-x8 | 1.140 (0.98-1.40; 4 of 204; 55 left out) | 1.126 (0.93-1.36; 5 of 63; 33 left out) | 0.997 (7) / 0.998 (0) |
| EPYC 9V74 (Zen 4 server) | host-tuned default | double, AVX-x4 | 1.118 (0.99-1.38; 2 of 191) | 1.142 (0.92-1.37; 5 of 63) | 1.000 (5) / 0.997 (4) |
| EPYC 9V74 (Zen 4 server) | x86-64 baseline | float, SSE1-x4 and SSE1-x8 | 1.140 (1.03-1.43; 0 of 204; 55 left out) | 1.128 (0.93-1.36; 6 of 63; 33 left out) | 1.000 (3) / 0.999 (6) |
| EPYC 9V74 (Zen 4 server) | x86-64 baseline | double, SSE2-x4 | 1.070 (0.91-1.22; 6 of 191) | 1.074 (0.96-1.22; 7 of 63) | 0.999 (1) / 1.001 (4) |
| EPYC 9V45 (Zen 5 server, 2 runners) | host-tuned default | float, SSE1-x4 | 1.078-1.080 (1.01-1.26; 0 of 191) | 1.137-1.142 (0.93-1.40; 2-4 of 63) | 1.077-1.079 (4-8) / 1.004-1.005 (7-15) |
| EPYC 9V45 (Zen 5 server, 2 runners) | host-tuned default | double, AVX-x4 | 1.094-1.097 (0.93-1.34; 12-13 of 191) | 1.182-1.183 (1.02-1.36; 0 of 63) | 0.986 (62-63) / 1.003 (8-10) |
| EPYC 9V45 (Zen 5 server, 2 runners) | x86-64 baseline | float, SSE1-x4 | 1.090-1.091 (0.95-1.30; 1-2 of 191) | 1.125-1.138 (0.97-1.39; 2 of 63) | 1.001-1.002 (2-4) / 0.999-1.000 (8-10) |
| EPYC 9V45 (Zen 5 server, 2 runners) | x86-64 baseline | double, SSE2-x4 | 1.085-1.087 (0.90-1.24; 0-5 of 191) | 1.095 (0.96-1.29; 3-5 of 63) | 0.997-0.998 (16) / 0.996-0.997 (13-26) |

The single-lane scalar rows have fewer changed complex cases (171), as only
the terminal radix-8 pass applies there; the rows with 8 lanes at every
size have more (238), as with 8 lanes more cases change their plan or
routing. In the float rows of builds that choose the lane count per size
(the default build as shipped on the Ryzen 7 5800HS, the EPYC 7763 and
9V74), the counts depend on which sizes run 8 lanes and, on the EPYC
runners, on the cases left out.

### Slower cases

Over all rows, 453 changed cases read below 0.97. The two lowest are one
case on the Ryzen 7 5800HS, float complex N = 128 in-place forward at 8
lanes: 0.813 in the build with 8 lanes at every size and 0.817 in the
default build as shipped, which runs this size at 8 lanes on that CPU. Most
of that loss is not the routing change: in the same two files the
operations whose code path is unchanged read 0.85 to 0.88 at that size,
and the same case reads 1.10 to 1.17 on the EPYC 7763 and 9V74 and 1.12 on
the Haswell Xeon with 8 lanes at every size (item 8). Then come the
Cortex-A53 at 0.820 (float complex forward, N = 30720; item 4), the Xeon's
4xScalar build at 0.832 (double real N = 512 in-place ordered forward), the
single-lane scalar rows with the terminal radix-8 pass forced on, from
0.836 (float complex N = 1536 in-place forward on the Xeon; item 5), and
the Neoverse-N2's double real in-place transforms at N = 128, from 0.847
(item 3). The 4xScalar build on the Xeon has 37 of 191 double complex
changed cases below 1.00, single cases down to 0.89 in double complex and
in float real, and in double real one case below 1.00 (the 0.832 above), at
geometric means of 1.059, 1.152 and 1.131; the survey has no breakdown of
these.

1. **Code placement on Zen 3, shown on an untouched function.**
   `pffftd_zreorder` on complex data (double precision, 4 lanes, host-tuned
   build) reads 0.80 to 0.85 at every size from 64 to about 20,000 on the
   Ryzen 7 5800HS (0.87 to 0.96 above), although its source is not part of
   the change. On a Ryzen 9 5950X the same pair of builds reads 0.82, 0.82,
   0.84 and 0.86 at N = 256, 1024, 4096 and 16384 (59 ns for the old build
   against 72 ns for the new at N = 256); with both builds compiled with
   `-falign-functions=64 -falign-loops=64` it reads 1.00, 1.00, 1.00 and
   0.99 (65 ns for both). The old build's loop happened to land in a fast
   position and the new build's in a slow one. This is the "other cases"
   figure of 0.978 with 44 cases below 0.97 in the table; without
   `zreorder` those cases are at 1.00. Samples, 12 pairs per case, with the
   forward transform and `zconvolve` as controls at 1.00:
   [default placement](results/pass-schedule-placement-zreorder-p0.csv.gz),
   [aligned](results/pass-schedule-placement-zreorder-p1.csv.gz). Single
   cases on Zen 3 move by such amounts with code placement alone, so a
   single case there says little.
2. **Complex N = 2560 and 7680 on Zen 3** (plans 5x4x4x8 and 5x3x4x4x8,
   4 lanes). On the Ryzen 7 5800HS the in-place forward transform reads 0.93
   (float) and 0.93 (double) at N = 2560 and 0.96 and 0.94 at N = 7680; at
   N = 2560 float backward, ordered backward, in-place backward and the
   pipeline read 0.93 to 0.97, while forward, ordered forward and the
   in-place ordered modes there gain 4 to 13%. The Haswell Xeon is faster in
   every one of these cases (2 to 25%). Not examined further.
3. **Real in-place transforms at a few sizes.** The parity of the pass
   count changes there, and the in-place first pass is a change to complex
   transforms only. On 4 lanes the unordered in-place real transforms read,
   at N = 128, 2048, 32768 and 524288: on the Neoverse-N2 in double
   precision 0.85 to 0.89, 0.87 to 0.90, 0.91 to 0.96 and 0.96 to 0.99 (all
   three runners; in float 0.98 to 1.05 at the same sizes); on the Haswell
   Xeon 0.96 for N = 2048 in-place forward in float (in the host-tuned and
   the x86-64 baseline build); on the Ryzen 7 5800HS 0.97 for N = 32768
   in-place backward and 0.96 for N = 524288 in-place ordered backward in
   float. The out-of-place modes at the same sizes gain. On the x86 runners
   of GitHub (item 6) the real changed cases below 0.97 on at least one
   runner are all in in-place modes: on the EPYC 9V74 float N = 65536 at 8
   lanes (in-place backward 0.93, in-place ordered backward 0.95 and 0.96;
   both builds), double N = 32768 (in-place backward 0.92 and in-place
   ordered backward 0.93 host-tuned; three in-place modes at 0.96 to 0.97
   baseline) and double N = 2048 in-place backward 0.96 (baseline); on the
   EPYC 9V45 float N = 524288 in-place forward and backward (0.94 and 0.93
   on one runner, 1.00 and 1.06 on the other; host-tuned) and double
   N = 32768 in-place backward (0.96 on one runner, 0.99 on the other;
   baseline); on the EPYC 7763 double N = 524288 in-place ordered forward
   (0.91 on one runner, 1.07 and 1.09 on the other two; host-tuned), and no
   real case in float or in the baseline build.
4. **Cortex-A53 (Raspberry Pi 3), NEON.** In float complex the sizes with a
   changed plan gain up to N = 2048 (geometric mean per size 1.06 to 1.17)
   and from 131072 up (1.03 to 1.05), and read 0.95 to 0.98 at N = 8192,
   24576, 30720 and 32768 (1.01 at 10240 and 40960); single cases go down to
   0.82 (forward at N = 30720). The pairs of one case scatter by about 5% on
   that board (governor `ondemand`, under-voltage and throttling flags set),
   so single cases mean little there. The Neoverse-N2 has no complex case
   below 1.00 in either precision. In double complex the board reads 1.052
   over the changed cases (0.87 to 1.25; 22 of 191 below 1.00): per size,
   the sizes with a changed plan read 1.08 to 1.10 from N = 96 to 640, 1.03
   to 1.05 from 1536 to 2560 and from 24576 to 40960, 1.02 to 1.03 at 131072
   and 524288, and 0.99 to 1.00 at N = 6144, 7680 and 10240 (1.04 at 8192).
   Its float real row reads 1.100 (0.92 to 1.28; 4 of 63 below 1.00).
5. **The single-lane scalar build** (no SIMD, one lane; not what CMake
   builds by default without SIMD, which is 4xScalar). With the terminal
   radix-8 pass forced on, complex transforms are slower at the sizes whose
   plan changes (geometric means 0.942 in float and 0.994 in double on the
   Xeon, 0.978 in float on the Neoverse-N2), while real transforms gain
   (1.152 in float and 1.130 in double on the Xeon, 1.081 in float on the
   Neoverse-N2). The switch is one per build, so the terminal radix-8 pass
   is off by default on that build. 4xScalar gains in both transform types
   and keeps the default.
6. **The x86 runners of GitHub (EPYC 7763, EPYC 9V74, EPYC 9V45): which
   code they compared.** The new library of these twelve files was built
   from a revision that has the pass code of this change (it differs from
   the merged sources in comments and in the defaults of other targets
   only) but other 8-lane size tables than `master`: a regenerated set that
   is not part of this change. Both builds dispatch at run time, so on the
   two CPUs that have a size table (EPYC 7763, EPYC 9V74) some sizes ran 4
   lanes in one build and 8 lanes in the other. Those cases are left out
   (55 of the 440 complex cases, and 22 of the 385 real cases on the EPYC
   7763 or 33 on the EPYC 9V74, per float file); the rows hold the cases in
   which both builds ran the same lane count, which is the lane count the
   merged code runs there, because its tables are `master`'s. At the sizes
   that are left out, the new passes were not timed on these machines at the
   lane count the merged code uses. The EPYC 9V45 is covered by no size
   table (4 lanes at every size) and double precision has no 8-lane code,
   so nothing is left out in those files. The slower changed complex cases
   on these runners (below 0.97 on at least one runner) are, all of them:
   in float none on the EPYC 7763 and 9V74 (386 and 408 cases over both
   builds), and on the EPYC 9V45 N = 131072 forward in the baseline build
   (1.03 on one runner, 0.95 on the other); in double N = 10240 in-place
   forward on the EPYC 7763 (0.96, 0.99, 0.96; host-tuned), five operations
   at N = 524288 on the EPYC 9V74 in the baseline build (0.91 to 0.96, its
   single runner), and on the EPYC 9V45 the pipeline at N = 524288 in the
   host-tuned build (0.93 on both runners) and cases that are below 0.97 on
   one runner and at 1.00 or above on the other (host-tuned: N = 131072
   in-place backward 0.96 / 1.14, N = 524288 backward 0.97 / 1.01;
   baseline: N = 524288 forward, ordered forward and pipeline 0.94 to 0.95
   on runner 2 against 1.00 to 1.05 on runner 1, N = 128 in-place ordered
   forward 1.14 / 0.90). The real ones are in item 3.
7. **EPYC 9V45 (Zen 5), host-tuned build: two effects on code the change
   does not touch.** GCC 13.3 on the runners does not know this CPU
   (`-march=native`). Both effects are the same on both runners of this
   model, and neither shows in the x86-64 baseline build on the same two
   machines or in the host-tuned build on the EPYC 9V74 and 7763.
   (a) `pffft_zreorder` on complex float data reads 1.27 to 1.71 on runner
   1 and 1.11 to 1.70 on runner 2 at the 40 sizes (geometric mean 1.61 and
   1.59), although its source is unchanged. It alone is the "other cases"
   mean of 1.079 and 1.077 of that row: the transforms with unchanged plan
   and routing read 0.999 on both runners.
   (b) In double precision the complex transforms whose plan and routing
   are unchanged read 0.979 and 0.978 in the geometric mean (169 cases),
   with 61 and 60 cases below 0.97, of which 59 and 57 at N >= 4096 (the
   lowest 0.90). Per size (geometric mean of the unchanged operations,
   runner 1 / runner 2): N = 4096: 0.969 / 0.978, 12288: 0.949 / 0.945,
   16384: 0.931 / 0.938, 20480: 0.948 / 0.952, 49152: 0.948 / 0.942,
   61440: 0.963 / 0.950, 65536: 0.935 / 0.955, 262144: 0.937 / 0.920.
   N = 5120, 15360 and 1048576 read 0.979 to 0.995, and every such size
   below 4096 reads 0.993 to 1.008. The baseline build on the same runners
   reads 0.991 to 1.017 at all these sizes, and the host-tuned build on the
   EPYC 9V74 0.991 to 1.003 at N >= 4096. The sizes whose plan changes gain
   on this build as well (1.094 and 1.097, 12 and 13 of the 191 cases below
   1.00, the lowest 0.93).
   The cause was not examined (no machine of this type was at hand); on
   Zen 3 an effect of this kind on the same function was the placement of
   the code (item 1).
8. **Float complex N = 128 at 8 lanes on the Ryzen 7 5800HS: the two lowest
   changed cases of the survey.** The plan (4x4 at 8 lanes) is unchanged at
   this size; only the in-place modes change their routing. In-place forward
   reads 0.813 in the build with 8 lanes at every size and 0.817 in the
   default build as shipped, which runs this size at 8 lanes on that CPU.
   In the same two files the operations whose code path is unchanged read:
   forward 0.85 and 0.88, ordered backward 0.87 and 0.87, pipeline 0.85 and
   0.86; in-place backward reads 1.22 in both and the in-place ordered
   modes 1.11 to 1.17. So most of the loss is not the routing change. The
   same case (in-place forward, N = 128, 8 lanes) reads 1.12 to 1.17 on the
   EPYC 7763 (three runners, both builds), 1.17 and 1.10 on the EPYC 9V74
   (host-tuned, baseline) and 1.12 on the Haswell Xeon with 8 lanes at every
   size. The unchanged forward transform at this size reads 1.06 to 1.08 in
   the host-tuned build and 0.88 to 0.89 in the baseline build on the EPYC
   7763 (all three runners), 0.98 to 1.02 on the EPYC 9V74 and the Xeon.
   Not examined further; item 1 shows how much code placement alone moves a
   single case on Zen 3.

### Not measured

- PowerPC (AltiVec/VSX) and ARMv7 only run the test suite, under emulation
  in CI; WebAssembly only runs the tests. MSVC builds and 32-bit x86 were
  not timed.
- No Intel CPU other than the Haswell-EP Xeon. Desktop Zen 3 (Vermeer),
  which has a size table of its own, only in the two code-placement files
  of the Ryzen 9 5950X (item 1).
- The Cortex-A53 in double precision real and its single-lane scalar
  build; the Neoverse-N2's single-lane scalar build in double precision.
- On the EPYC 7763 and 9V74, the float cases that are left out (item 6):
  at those sizes the new passes were not timed on these machines at the
  lane count the merged code uses.

### Size tables

The 8-lane size tables ([pffft_wide_sizes.h](../../src/pffft_wide_sizes.h)),
[wide_size_table.py](wide_size_table.py) and the sweeps behind them are not
changed: they were measured with the earlier passes. Every size therefore
runs the lane count it ran before, so no size can get slower through a
table; sizes at which the 8-lane code would now newly pass the rule stay on
4 lanes until the tables are measured again.

### Benchmark tooling

- `run_matrix.py`: `--seed` sets the order of the cases and of the builds
  within a repeat (independent invocations, such as the parts of one
  comparison, need different seeds); every invocation appends a record to
  `<output>.invocations` (start, end, rows before and after, seed, SHA-256
  of each driver binary); `--resume` is safe after a killed invocation (its
  rows move to `<output>.discarded` and are measured again);
  `--perf-events` runs each sample under `perf stat` (counts for the whole
  driver process); `--idle-sibling` rejects and repeats a sample during
  which an SMT sibling of the measuring CPU was busy; `--self-test` checks
  the parsers, the sibling decision, the order and the records without
  running a benchmark (CTest `run_matrix_selftest`, on Linux hosts).
- `bench_hotloops.cpp` checks its size argument with `pffft_is_valid_size()`
  (`pffftd_is_valid_size()` in double) instead of creating a setup first.
  Creating and destroying a setup before the driver allocated its buffers
  changed where glibc placed them: consecutive heap blocks staggered by 64
  or 128 bytes within their pages instead of separate mappings at one page
  offset. At sizes with a large power-of-two factor that changed timings by
  several percent in either direction; 4K aliasing between loads and
  earlier stores is the likely mechanism. The driver's buffers now lie where
  those of a program that allocates them first do.
- `mca_predict.py` reads the inferred table of `pffft_wide_sizes.h` under its
  current array names again, and has a `--self-test` for that reader (CTest
  `mca_predict_selftest`).
