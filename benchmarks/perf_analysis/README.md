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
if, in each of two independent sweeps, all eight transform modes and the
FFT-multiply-IFFT pipeline were at least 1.10x as fast in the median with
non-overlapping interquartile ranges, and `pffft_zreorder` and
`pffft_zconvolve` were at least 0.97x. The sweeps
([1](results/wide-sweep1.csv.gz), [2](results/wide-sweep2.csv.gz), 26532
samples each) cover all 201 complex and 167 real sizes from 64 to 2^20 that
the 8-lane code supports, all eleven operations, 3 samples per case and
build in randomized interleaved order, with the `native4` and `wide` builds
above. The other settings are those of the measurement above, except that
the host-load limit was raised to 4 busy CPUs (sweep 1 started at 2.5) to
make progress on a shared host; the IQR condition guards against the extra
noise. Per-size verdicts: [wide-size-table.txt](results/wide-size-table.txt).

Enabled: 192 complex and 152 real sizes. There, the slowest of the nine
gated modes per size was 1.19x (complex, N = 64) to 1.61x, and 1.12x (real,
N = 128, in-place) to 1.61x; over all modes and sizes the median speedup
was 1.48x (complex) and 1.41x (real). Spectral multiplication is up to 2x
faster at small N and about equal once it is memory bound (0.97-1.28x from
N = 16384); real `pffft_zreorder` is 0.97x (N = 128) to 2x.

Not enabled: real N = 256 (in-place backward 1.10x in both sweeps, just
under the threshold), and 9 complex and 14 real sizes from 20480 to 663552
where `pffft_zconvolve` measured 0.95-0.97x in at least one sweep, although
the transforms there were 1.23-1.65x as fast. The kernel is the 4-lane loop
on 256-bit vectors and memory bound at these sizes; the conservative rule
keeps the 4-lane code rather than trade a few percent in one operation.

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

Against the controlled FFTW builds (bench_pffft, MFlops): complex unordered
PFFFT now matches FFTW AVX2 at N = 512..2048 (91465 vs 90781 at 1024) and is
ahead from N = 8192; ordered complex output is still 15-35% behind FFTW AVX2
at N = 64..4096; real transforms are 1.2-1.8x as fast as FFTW AVX2.

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

### Tried and rejected

| Candidate | Result |
| --- | --- |
| Radix-8 first pass replacing the radix-2 and one radix-4 pass (8-lane complex, N/8 = 2·4^k) | 0.80-1.19x, no consistent gain ([samples](results/wide-radix8-rejected.csv.gz)); consistent with the arm64 radix-8 result in 2ce8a89 |
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
- `check_isa_baseline.py` (a ctest in baseline x86-64 Linux builds): no VEX or
  EVEX instruction outside the 8-lane functions, checked for GCC and Clang
  with and without LTO.
- `tests/run_emulated_cpu_tests.sh` (CI job `x86_cpu_fallback`): static
  baseline binaries pass under QEMU emulating Nehalem and SandyBridge without
  selecting the backend (QEMU raises SIGILL for AVX2 there), and use and pass
  it on Haswell.

### Limitations

- Measured on one CPU model; other AVX2 CPUs keep the 4-lane code unless
  built with `PFFFT_WIDE_UNKNOWN_CPUS=ON`. Adding a CPU class to
  `pffft_x86_wide_measured()` needs the same sweep on that CPU.
- Not built on Windows (MSVC lacks per-function target selection, MinGW GCC
  cannot realign the stack for 256-bit spills, clang-cl is untested).
- Double precision stays on its existing 4-lane AVX code; AVX-512 is not used.
- Buffers at 16 mod 32 bytes are correct but lose part of the gain
  (prototype: 1.15-1.67x instead of 1.4-1.9x).
