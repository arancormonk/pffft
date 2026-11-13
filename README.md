
---

# PFFFT: a pretty fast FFT and fast convolution with PFFASTCONV

---

<!-- toc -->

- [Brief Description](#brief-description)
- [Why does it exist?](#why-does-it-exist)
- [CMake](#cmake)
- [Performance and architecture notes](#performance-and-architecture-notes)
- [Using pffft in your CMake project](#using-pffft-in-your-cmake-project)
- [History / Origin / Changes](#history--origin--changes)
- [Comparison with other FFTs](#comparison-with-other-ffts)
- [Dependencies / Required Linux packages](#dependencies--required-linux-packages)
- [Benchmarks and results](#benchmarks-and-results)
- [Appendix: upstream commit map](#appendix-upstream-commit-map)

<!-- tocstop -->

---

## Brief description:

PFFFT does 1D Fast Fourier Transforms, of single precision real and
complex vectors. It tries do it fast, it tries to be correct, and it
tries to be small. Computations do take advantage of SSE1/AVX/AVX2 instructions
on x86 cpus, Altivec (float) and VSX (double) on powerpc cpus, NEON on ARM cpus
(including Apple Silicon), and WASM SIMD on WebAssembly. The license is BSD-like.

The SIMD backend is picked at compile time from what the compiler targets:
NEON whenever it is enabled (always on AArch64; on 32-bit ARM with e.g.
`-mfpu=neon`), VSX with e.g. `-mcpu=power7` (the ppc64le default), WASM SIMD
with `-msimd128`.

PFFFT is a fork of [Julien Pommier's library on bitbucket](https://bitbucket.org/jpommier/pffft/)
with some changes and additions. This repository,
[arancormonk/pffft](https://github.com/arancormonk/pffft), is maintained by
arancormonk as a fork of [marton78/pffft](https://github.com/marton78/pffft);
see [History / Origin / Changes](#history--origin--changes) for what it adds.


PFFASTCONV does fast convolution (FIR filtering), of single precision 
real vectors, utilizing the PFFFT library. The license is BSD-like.

PFDSP contains a few other signal processing functions.
Currently, mixing and carrier generation functions are contained.
It is work in progress - also the API!
The fast convolution from PFFASTCONV might get merged into PFDSP.


## Why does it exist:

I (Julien Pommier) was in search of a good performing FFT library ,
preferably very small and with a very liberal license.

When one says "fft library", FFTW ("Fastest Fourier Transform in the
West") is probably the first name that comes to mind -- I guess that
99% of open-source projects that need a FFT do use FFTW, and are happy
with it. However, it is quite a large library , which does everything
fft related (2d transforms, 3d transforms, other transformations such
as discrete cosine , or fast hartley). And it is licensed under the
GNU GPL , which means that it cannot be used in non open-source
products.

An alternative to FFTW that is really small, is the venerable FFTPACK
v4, which is available on NETLIB. A more recent version (v5) exists,
but it is larger as it deals with multi-dimensional transforms. This
is a library that is written in FORTRAN 77, a language that is now
considered as a bit antiquated by many. FFTPACKv4 was written in 1985,
by Dr Paul Swarztrauber of NCAR, more than 25 years ago ! And despite
its age, benchmarks show it that it still a very good performing FFT
library, see for example the 1d single precision benchmarks
[here](http://www.fftw.org/speed/opteron-2.2GHz-32bit/). It is however not
competitive with the fastest ones, such as FFTW, Intel MKL, AMD ACML,
Apple vDSP. The reason for that is that those libraries do take
advantage of the SSE SIMD instructions available on Intel CPUs,
available since the days of the Pentium III. These instructions deal
with small vectors of 4 floats at a time, instead of a single float
for a traditionnal FPU, so when using these instructions one may expect
a 4-fold performance improvement.

The idea was to take this fortran fftpack v4 code, translate to C,
modify it to deal with those SSE instructions, and check that the
final performance is not completely ridiculous when compared to other
SIMD FFT libraries. Translation to C was performed with [f2c](
http://www.netlib.org/f2c/). The resulting file was a bit edited in
order to remove the thousands of gotos that were introduced by
f2c. You will find the fftpack.h and fftpack.c sources in the
repository, this a complete translation of [fftpack](
http://www.netlib.org/fftpack/), with the discrete cosine transform
and the test program. There is no license information in the netlib
repository, but it was confirmed to me by the fftpack v5 curators that
the [same terms do apply to fftpack v4]
(http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html). This is a
"BSD-like" license, it is compatible with proprietary projects.

Adapting fftpack to deal with the SIMD 4-element vectors instead of
scalar single precision numbers was more complex than I originally
thought, especially with the real transforms, and I ended up writing
more code than I planned..


## The code:

### Good old C:
The FFT API is very very simple, just make sure that you read the comments in `pffft.h`.

The Fast convolution's API is also very simple, just make sure that you read the comments 
in `pffastconv.h`.

### C++:
A simple C++ wrapper is available in `pffft.hpp`.

### Git:
This archive's source can be downloaded with git (without the submodules):
```
git clone https://github.com/arancormonk/pffft.git
```

### Only two files?:
_"Only two files, in good old C, pffft.c and pffft.h"_

This statement does **NO LONGER** hold!

With new functionality and support for AVX, there was need to restructure the sources.
But you can compile and link **pffft** as a static library.


## CMake:
There's now CMake support to build the static libraries `libPFFFT.a`
and `libPFFASTCONV.a` from the source files. The additional
`libFFTPACK.a` library is built only when tests or benchmarks are enabled.

There are several CMake options to modify library size and optimization.
You can explore all available options with `cmake-gui` or `ccmake`,
the console version - after having installed (on Debian/Ubuntu Linux) one of
```
sudo apt-get install cmake-qt-gui
sudo apt-get install cmake-curses-gui
```

Some of the options:
* `PFFFT_USE_TYPE_FLOAT` to activate single precision 'float' (default: ON)
* `PFFFT_USE_TYPE_DOUBLE` to activate 'double' precision float (default: ON)
* `PFFFT_USE_SIMD` to use SIMD (SSE/AVX/NEON/ALTIVEC/WASM SIMD) CPU features? (default: ON)
* `PFFFT_USE_SCALAR_VECT` to use 4-element vector scalar operations (if no other SIMD) (default: ON)
* `PFFFT_USE_WASM_RELAXED_SIMD` to use WebAssembly Relaxed SIMD multiply-add instructions, which may be fused when supported by the runtime (default: OFF)

Options can be passed to `cmake` at command line, e.g.
```
cmake -DPFFFT_USE_TYPE_FLOAT=OFF -DPFFFT_USE_TYPE_DOUBLE=ON
```

My Linux distribution defaults to GCC. With installed CLANG and the bash shell, you can use it with
```
mkdir build
cd build
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -DCMAKE_BUILD_TYPE=Debug ../
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=~ ../
ccmake .                          # or: cmake-gui .
cmake --build .                   # or simply: make
ctest                             # to execute some tests - including benchmarks
cmake --build . --target install  # or simply: [sudo] make install
```

With MSVC on Windows, you need some different options. Following ones to build a 64-bit Release with Visual Studio 2019:
```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
ctest -C Release
```

see [https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html#visual-studio-generators](https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html#visual-studio-generators)

With [Emscripten](https://emscripten.org/) for WebAssembly (requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)):

```sh
mkdir build
cd build
emcmake cmake ..
cmake --build .
ctest
```

WASM SIMD is enabled automatically when building with Emscripten; see
[Performance and architecture notes](#performance-and-architecture-notes) for
the Relaxed SIMD option and the non-CMake build flags.

## Performance and architecture notes

Compiler- and architecture-specific tips for getting the best performance out
of pffft.

### Choose a Release configuration

A plain top-level configure is tuned for the build host: `TARGET_C_ARCH` and
`TARGET_CXX_ARCH` default to `native` (`-march=native`) with GCC and Clang. MSVC
and clang-cl have no `/arch:native`, so configure probes the build host's CPU and
picks the highest of `AVX512`, `AVX2` and `AVX` it can run. Such a build requires a
CPU compatible with the build host. The only runtime ISA dispatch is the choice
of the 8-lane AVX2 float implementation per setup, see
[below](#x86-8-lane-avx2-float-transforms).

The default stays at the compiler's baseline ISA (`none`) when host tuning would
be wrong: when pffft is built as a dependency (`add_subdirectory`,
`FetchContent`), when cross compiling (Android, iOS and Emscripten included),
when `CMAKE_OSX_ARCHITECTURES` names several or a non-host architecture, when
`CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` already select an ISA (`-march=`, `-mcpu=`,
`/arch:`, as distribution build flags usually do), or when the compiler rejects
`-march=native`. CMake prints which default it chose and why.

With CMake 3.21 or newer and Ninja, the repository provides these presets:

| Preset | Intended use |
| --- | --- |
| `release` | Packages: compiler baseline ISA, no additional architecture flags |
| `release-native` | GCC/Clang build for the current host, including its available AVX/FMA features |
| `release-haswell` | GCC/Clang x86 package explicitly requiring the Haswell instruction set |
| `release-native-unfused` | Native build with explicit FMA and compiler contraction disabled |

```sh
cmake --preset release-native
cmake --build --preset release-native --parallel
ctest --preset release-native
```

Build binaries meant for other machines with the `release` preset, or pass
`-DTARGET_C_ARCH=none -DTARGET_CXX_ARCH=none`; alternatively distribute a
separately labelled `release-haswell` package to compatible x86 machines. `none`
adds no flags, so it builds for the compiler's default ISA, which is not
baseline x86-64 everywhere: GCC on GitHub's `ubuntu-26.04` runners emits
x86-64-v3 (AVX2, FMA) code by default. Configure reports it when the default is
above baseline; with GCC or Clang on x86-64,
`-DTARGET_C_ARCH=x86-64 -DTARGET_CXX_ARCH=x86-64` selects the baseline. On
MSVC, select a level explicitly with `-DTARGET_C_ARCH=AVX2 -DTARGET_CXX_ARCH=AVX2`
(or `SSE2`, `AVX`, `AVX512`). The archives uploaded by this repository's CI use
the baseline ISA, except the MSVC builds labelled `build_sse2`, `build_avx` and
`build_avx2`.

On a Ryzen 5950X, selecting the native backend made a 1024-point double real
forward FFT 2.08 times as fast as the baseline SSE2 build. The float transforms
are built four-wide (SSE) but can use the 8-lane AVX2 implementation described
below; the gain depends on precision, size and workload. See the
[performance analysis](benchmarks/perf_analysis/README.md) for measurements.

Ordered complex transforms fuse spectrum reordering into finalization and
preprocessing using the existing SIMD backends. Recursive mixers explicitly
vectorize independent oscillator lanes where the mixer SIMD backend is
available. Their recurrence also handles the formerly unstable rates near
an odd multiple of pi per lane step, including `rate=0.125` for eight lanes.

### x86: enable FMA

On x86 the fused multiply-add paths need an architecture with FMA3. A
host-tuned default build on an FMA-capable machine gets them; baseline-ISA
builds (`TARGET_C_ARCH=none`, the `release` preset, packages) do not.
`VMADD`/`VMSUB` -- used by the complex multiply at the heart of
`pffft_zconvolve*()` and of every radix codelet -- lower to a real FMA
instruction only when the compiler guarantees FMA3:

| compiler | how to request it explicitly |
| --- | --- |
| gcc / clang | `-DTARGET_C_ARCH=haswell` (or a later `-march` value) |
| MSVC | `-DTARGET_C_ARCH=AVX2` |

Without it, each fused operation costs a separate multiply plus add. Measured
on the `pffft_zconvolve_accumulate()` inner loop, identical for gcc 12 and
clang 14, x86-64 and i686, float (SSE) and double (AVX): **20 floating point
operations per iteration instead of 12**.

```
cmake -DTARGET_C_ARCH=haswell -DTARGET_CXX_ARCH=haswell ..
```

Host-tuned builds get FMA when the build host has it. The baseline ISA used for
packages does not on purpose: FMA3 requires an Intel Haswell (2013) or AMD
Piledriver CPU, so requiring it would break older hardware. `cmake` prints a
reminder when the selected x86 architecture has no FMA.

Each FMA rounds once where a separate multiply and add round twice, so final
transform results can differ. Fewer instructions do not guarantee a speedup
for every size. `PFFFT_DISABLE_FMA` controls explicit FMA in both PFFFT and
PFDSP; also disable compiler contraction when unfused arithmetic is required.

### x86: 8-lane AVX2 float transforms

pffft splits an N-point transform into as many interleaved sub-sequences as a
vector has lanes, runs the FFTPACK passes on all of them at once, and combines
them across the lanes at the end. On x86-64 the library also contains this
algorithm with 8 lanes on AVX2 (`src/pffft_wide_impl.h`), compiled with GCC or
Clang target attributes next to the 4-lane SSE code in `pffft.c`, and with
MSVC as a translation unit of its own (see [MSVC](#msvc-and-the-8-lane-code)).
`pffft_new_setup()` chooses it when

* the build has it: x86-64 with GCC 5 or later (on Windows MinGW-w64 GCC 13
  or later, optimized builds only: GCC cannot align 32-byte stack slots
  there at -O0), Clang 10 or later (on Windows, including clang-cl, 19 or later)
  or MSVC from Visual Studio 2022 17.14, SSE float backend,
  `PFFFT_ENABLE_WIDE_DISPATCH` (default `ON`);
* the CPU and OS support AVX2 and FMA (`PFFFT_DISABLE_FMA` drops the FMA
  requirement), and every other extension the compiled 8-lane code may use,
  checked with CPUID/XGETBV on every call; the instructions only run after
  that check, so baseline-ISA builds stay safe on older CPUs;
* the CPU model is one the 8-lane code was measured to be faster on (AMD Zen 3
  and Zen 4 so far); a build can opt in to CPU models that were not measured, see
  [CPUs that were not measured](#cpus-that-were-not-measured);
* N is a multiple of 64 (complex) or 128 (real) with N/8 = 2^a 3^b 5^c, and
  is one of the sizes measured faster on that CPU (`src/pffft_wide_sizes.h`).

Otherwise the setup uses the 4-lane code, so every size accepted before is still
accepted. `pffft_setup_simd_size(setup)` reports the lane count a setup uses.
Ordered output is identical in layout (and equal to rounding); the unordered
("most efficient") layout differs between the two, so unordered spectra only
work with setups of the same N, transform and process, as `pffft.h` documents.
Buffers still only need 16-byte alignment; 32-byte aligned buffers (as
`pffft_aligned_malloc()` returns) are faster.

Each measured CPU model has its own table of the sizes where, in every sweep
over all such sizes up to 2^20, every transform mode was at least 1.10x as
fast and spectral multiplication and reordering kept at least 0.97x of their
speed: Zen 3 desktop parts (Vermeer, two sweeps on a Ryzen 9 5950X) at 192
of 201 complex and 152 of 167 real sizes, Zen 3 server parts (Milan, seven
sweeps on EPYC 7763 virtual machines of GitHub's runners) at 143 and 115,
Zen 4 (EPYC 9V74, three runner sweeps) at 186 and 163; the Zen 3 APUs, not
measured, use the 137 and 105 sizes that passed on both Zen 3 models. On the
5950X float transforms then run 1.19-1.91x (complex) and 1.12-1.97x (real)
as fast as with the 4-lane code, in every transform mode; for example a
1024-point complex forward FFT takes 589 instead of 980 ns and a 1024-point
real ordered forward FFT 334 instead of 471 ns. Against FFTW 3.3.10's AVX2
build with `FFTW_MEASURE` plans (a baseline pffft build on the 5950X), real
transforms are 1.3-1.8x as fast at every power of two from 32 to 262144
except 256 (0.92x), unordered complex transforms 0.97-1.15x at N = 256..2048
and 1.3-1.6x from 8192, but FFTW stays ahead at N = 32..128, 4096 and most
non-power-of-two complex sizes (0.54-0.91x); see the [performance
analysis](benchmarks/perf_analysis/README.md#other-fft-libraries).
Independently of the lane count, complex transforms now deinterleave their
input in the first FFTPACK pass and interleave their output in the last one,
which makes the 4-lane complex transforms (float and double) up to 18%
faster, by a median of 6-7% out of place, and in place up to 18% at most
sizes with an even pass count; see the [performance
analysis](benchmarks/perf_analysis/README.md#8-lane-avx2-float-backend-and-fused-layout-passes)
for the method, all results, the exceptions and the limitations.

#### MSVC and the 8-lane code

MSVC has no per-function target selection, and its AVX2 code is markedly
slower without `/arch:AVX2` (a median 1.36x over the 4-lane code, against
1.58x with the 8-lane code compiled for AVX2, in the measurement of the
performance analysis). So with MSVC the 8-lane
code is its own translation unit, `src/pffft_wide_tu.c`, compiled with
`/arch:AVX2`; `pffft.c` stays a baseline build and only calls into it after
the CPU check. The CMake build does this for MSVC x64. A project that builds
pffft with its own build system and MSVC adds `src/pffft_wide_tu.c` to the
sources, compiles that file with `/arch:AVX2` and defines `PFFFT_WIDE_TU` for
the library's sources:

```cmake
target_sources(mylib PRIVATE pffft/src/pffft_wide_tu.c)
set_source_files_properties(pffft/src/pffft_wide_tu.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/GL-")
target_compile_definitions(mylib PRIVATE PFFFT_WIDE_TU)
```

Without it, an MSVC build keeps the 4-lane code (unless the whole build
targets AVX2). As `/arch:AVX2` may also emit BMI and related instructions,
the CPU check then also requires BMI1, BMI2, LZCNT, MOVBE and F16C, which
every AVX2 CPU in the tables has. GCC, Clang and clang-cl need nothing extra
(`pffft_wide_tu.c` compiles to nothing there).

#### CPUs that were not measured

By default only measured CPU models use the 8-lane code. With
`-DPFFFT_WIDE_INFERRED=ON` (or `PFFFT_WIDE_INFERRED` defined for `pffft.c`)
it is also used on CPUs whose documented 256-bit floating-point datapaths are
full width and that have no AVX frequency licences: AMD Zen 2, Zen 3 Badami
and Zen 5, and Intel's client big cores from Haswell to Rocket Lake
plus P-core-only Alder Lake-S and Raptor Lake-S parts, all by an explicit
list of CPUID family/model ranges (`src/pffft_cpu_x86.h`). They use a
stricter size table (every mode at least 1.25x in every measured sweep: 81
complex, 67 real sizes), and only sizes whose buffers fit in the core's L2
cache, as the cache leaves of CPUID report it. Everything else stays 4-lane: AVX2
cores that split 256-bit operations (Zen, Zen+, Hygon, Excavator, Intel Atom
cores), hybrid parts (a thread may run on their E-cores), Intel server
parts (AVX frequency licences), unknown models and future families. This is
an inference, not a measurement: a static throughput model (llvm-mca)
failed its pre-registered calibration against the Zen 3 measurements and
could not tell 128-bit and 256-bit datapaths apart, so it is off by default.
`-DPFFFT_WIDE_UNKNOWN_CPUS=ON` uses the 8-lane code with the measured table
on every AVX2 CPU. Measuring more CPUs moves them to the measured class, see
the [performance analysis](benchmarks/perf_analysis/README.md#cpu-classes).

### Pass schedule: terminal radix-8 pass and in-place first pass

Two changes to the FFTPACK pass schedule (`src/pffft_pass_config.h`) save a
sweep over the data where they apply:

* the terminal radix-8 pass (`PFFFT_TERMINAL_RADIX8`): a plan has no
  stand-alone radix-2 pass any more; where it had a radix-2 and a radix-4
  pass, the pair is one twiddle-free radix-8 pass at the end (complex
  factors 2, ..., 4 become ..., 8; real power-of-two factors 2, 4, ..., 4
  become 4, ..., 8);
* the in-place first pass (`PFFFT_INPLACE_FIRST_PASS`): on every 4-lane
  backend and the 8-lane AVX2 one, a complex transform runs its first pass
  in place where the parity of the pass count would otherwise cost an
  in-place transform an extra sweep to deinterleave or interleave its data.
  Real transforms are unchanged.

Both are on by default on every target and backend, with one exception: on
the single-lane scalar build (`-DPFFFT_USE_SIMD=OFF
-DPFFFT_USE_SCALAR_VECT=OFF`) the terminal radix-8 pass is off by default,
because complex transforms measured slower with it there (real transforms
gained, but the switch is one per build). The 4xScalar build, which CMake
builds by default without SIMD, keeps both. `-DPFFFT_TERMINAL_RADIX8=OFF`
and `-DPFFFT_INPLACE_FIRST_PASS=OFF` switch them off, see [Build control
options](#build-control-options). At the sizes whose plan changes, results
differ in the last bits from those of an earlier build: the accuracy tests
pass, a bit-exact comparison against an earlier build would not. The 8-lane
size tables were measured with the earlier passes and are unchanged, so
every size keeps the lane count it had.

There is no exhaustive per-size acceptance run behind this change; it was
merged on a survey of seven CPU models: old against new, built and measured
on each machine, 5 interleaved pairs per case, 40 complex sizes from 64 and
35 real sizes from 128 up to 1,048,576, all eleven operations. Each cell
gives the ratio old time / new time (above 1 the new code is faster) over
the changed cases (the sizes and operations whose plan or buffer routing
changes): the geometric mean, the lowest and highest single case and how
many cases are below 1.00; where several runners of one CPU model were
measured, the range of their figures. The new library on the x86 runners of
GitHub (EPYC 7763, 9V74, 9V45) had other 8-lane size tables than `master`;
on the EPYC 7763 and 9V74, which have a size table, the float cases in
which the two builds ran different lane counts are left out (55 of the 440
complex and 22 or 33 of the 385 real cases). The other cases are unchanged
(geometric means 0.992 to 1.005), except 0.978 for the Ryzen 7 5800HS in
double complex, an effect of code placement, and, with the host-tuned build
on the EPYC 9V45, 1.077 to 1.079 in float complex and 0.986 in double
complex (see below). The rows of the single-lane scalar build (terminal
radix-8 pass forced on) and of the builds with 8 lanes at every size are
in the analysis document only.

| CPU | build | precision, backend | complex | real |
| --- | --- | --- | --- | --- |
| Ryzen 7 5800HS (Zen 3) | host-tuned, 4 lanes, no dispatch | float, SSE1-x4 | 1.071 (0.93-1.23; 13 of 191) | 1.084 (0.96-1.27; 10 of 63) |
| Ryzen 7 5800HS (Zen 3) | host-tuned, 4 lanes, no dispatch | double, AVX-x4 | 1.090 (0.93-1.40; 7 of 191) | 1.135 (0.98-1.35; 3 of 63) |
| Ryzen 7 5800HS (Zen 3) | host-tuned default, as shipped (Zen 3 size table) | float, SSE1-x4 and SSE1-x8 | 1.106 (0.82-1.38; 3 of 206) | 1.122 (0.98-1.32; 2 of 45) |
| Xeon E5-2697 v3 (Haswell-EP) | host-tuned default | float, SSE1-x4 | 1.136 (1.01-1.27; 0 of 191) | 1.148 (0.96-1.35; 1 of 63) |
| Xeon E5-2697 v3 (Haswell-EP) | host-tuned default | double, AVX-x4 | 1.146 (0.97-1.28; 1 of 191) | 1.149 (1.03-1.35; 0 of 63) |
| Xeon E5-2697 v3 (Haswell-EP) | x86-64 baseline | float, SSE1-x4 | 1.116 (1.01-1.23; 0 of 191) | 1.124 (0.96-1.34; 1 of 63) |
| Xeon E5-2697 v3 (Haswell-EP) | x86-64 baseline | double, SSE2-x4 | 1.106 (0.99-1.28; 1 of 191) | 1.109 (0.98-1.31; 2 of 63) |
| Xeon E5-2697 v3 (Haswell-EP) | 4xScalar (default without SIMD) | float, 4xScalar-x4 | 1.070 (0.95-1.24; 14 of 191) | 1.152 (0.89-1.42; 3 of 63) |
| Xeon E5-2697 v3 (Haswell-EP) | 4xScalar (default without SIMD) | double, 4xScalar-x4 | 1.059 (0.89-1.26; 37 of 191) | 1.131 (0.83-1.41; 1 of 63) |
| Cortex-A53 (Raspberry Pi 3) | default (NEON) | float, NEON-x4 | 1.056 (0.82-1.26; 38 of 191) | 1.100 (0.92-1.28; 4 of 63) |
| Cortex-A53 (Raspberry Pi 3) | default (NEON) | double, NEON-x4 | 1.052 (0.87-1.25; 22 of 191) | not measured |
| Neoverse-N2 (3 runners) | default (NEON) | float, NEON-x4 | 1.078 (1.03-1.19; 0 of 191) | 1.068-1.069 (0.98-1.16; 3-4 of 63) |
| Neoverse-N2 (3 runners) | default (NEON) | double, NEON-x4 | 1.076-1.077 (1.00-1.19; 0 of 191) | 1.058 (0.85-1.27; 10 of 63) |
| EPYC 7763 (Zen 3 server, 3 runners) | host-tuned default | float, SSE1-x4 and SSE1-x8 | 1.134-1.137 (1.00-1.43; 0-1 of 193) | 1.132-1.137 (0.98-1.34; 1-4 of 45) |
| EPYC 7763 (Zen 3 server, 3 runners) | host-tuned default | double, AVX-x4 | 1.107-1.117 (0.96-1.40; 2 of 191) | 1.162-1.163 (0.91-1.37; 2 of 63) |
| EPYC 7763 (Zen 3 server, 3 runners) | x86-64 baseline | float, SSE1-x4 and SSE1-x8 | 1.134-1.137 (1.01-1.38; 0 of 193) | 1.130-1.131 (0.97-1.34; 0-2 of 45) |
| EPYC 7763 (Zen 3 server, 3 runners) | x86-64 baseline | double, SSE2-x4 | 1.078-1.083 (0.98-1.23; 0-1 of 191) | 1.103-1.118 (0.97-1.39; 2-3 of 63) |
| EPYC 9V74 (Zen 4 server) | host-tuned default | float, SSE1-x4 and SSE1-x8 | 1.140 (0.98-1.40; 4 of 204) | 1.126 (0.93-1.36; 5 of 63) |
| EPYC 9V74 (Zen 4 server) | host-tuned default | double, AVX-x4 | 1.118 (0.99-1.38; 2 of 191) | 1.142 (0.92-1.37; 5 of 63) |
| EPYC 9V74 (Zen 4 server) | x86-64 baseline | float, SSE1-x4 and SSE1-x8 | 1.140 (1.03-1.43; 0 of 204) | 1.128 (0.93-1.36; 6 of 63) |
| EPYC 9V74 (Zen 4 server) | x86-64 baseline | double, SSE2-x4 | 1.070 (0.91-1.22; 6 of 191) | 1.074 (0.96-1.22; 7 of 63) |
| EPYC 9V45 (Zen 5 server, 2 runners) | host-tuned default | float, SSE1-x4 | 1.078-1.080 (1.01-1.26; 0 of 191) | 1.137-1.142 (0.93-1.40; 2-4 of 63) |
| EPYC 9V45 (Zen 5 server, 2 runners) | host-tuned default | double, AVX-x4 | 1.094-1.097 (0.93-1.34; 12-13 of 191) | 1.182-1.183 (1.02-1.36; 0 of 63) |
| EPYC 9V45 (Zen 5 server, 2 runners) | x86-64 baseline | float, SSE1-x4 | 1.090-1.091 (0.95-1.30; 1-2 of 191) | 1.125-1.138 (0.97-1.39; 2 of 63) |
| EPYC 9V45 (Zen 5 server, 2 runners) | x86-64 baseline | double, SSE2-x4 | 1.085-1.087 (0.90-1.24; 0-5 of 191) | 1.095 (0.96-1.29; 3-5 of 63) |

On Zen 3, code placement alone moves single cases: `pffftd_zreorder`, whose
source did not change, read 0.80 to 0.85 at sizes up to about 20,000 on the
Ryzen 7 5800HS, and on a Ryzen 9 5950X 0.82 to 0.86 between the two builds
but 0.99 to 1.00 with both compiled with `-falign-functions=64
-falign-loops=64`.
On the EPYC 9V45 (Zen 5) the host-tuned build moves code the change does
not touch in both directions: the double-precision complex transforms whose
plan and routing are unchanged read 0.979 and 0.978 in the geometric mean
on its two runners, per size down to 0.920 at N >= 4096, and
`pffft_zreorder` in float reads 1.11 to 1.71; neither shows in the x86-64
baseline build on the same runners, and the cause was not examined.
The slower cases (among them the real in-place transforms at N = 2048 on 4
lanes and the Cortex-A53 at middle sizes), what was not measured (PowerPC,
ARMv7 and WebAssembly only run the tests; MSVC builds, 32-bit x86 and
Intel CPUs other than the Haswell-EP Xeon were not timed), the method and
the raw samples are in the [performance
analysis](benchmarks/perf_analysis/README.md#terminal-radix-8-pass-and-in-place-first-pass).

### ARMv7 (32-bit ARM): prefer clang over gcc

On 32-bit ARM, **gcc emits very inefficient assembly**, with a lot more
operations that clang does. Measured over the whole float translation unit,
both at `-O3 -march=armv7-a -mfpu=neon`:

| | 64-bit `vldr`/`vstr` | 128-bit `vld1`/`vst1` |
| --- | --- | --- |
| gcc 12.2 | 1600 | 344 |
| clang 14 | 181 | 797 |

GCC's bad code generation affects roughly 19 functions, including the hot ones
(`pffft_real_finalize`, `pffft_real_preprocess`, `pffft_cplx_finalize`,
`pffft_zreorder`, `passf4_ps`, `radf4_ps`, ...). This caveat applies to all versions
including GCC 16.1 (newest at time of writing), so this is not about old toolchains.

There is a GCC [ticket](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=43725)
reporting this since 2010 and it hasn't been fixed since. No flag changes it
(fourteen combinations were tried, including `-mtune=cortex-a9/a15`, `-mcpu=...`,
`-mfpu=neon-vfpv4`, `-mvectorize-with-neon-quad`, `-m[no-]unaligned-access`, `-O2`,
and the Raspberry Pi 3/4 presets). It doesn't seem like this problem is ever going
to be fixed, see [this comment](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=43725#c1)
by the ARM maintainer. So we recommend to **use Clang when targeting ARMv7**.

`cmake` prints a reminder when it sees gcc targeting ARMv7. Note that this applies
only to 32-bit ARM, on AArch64 both compilers are fine.

### WebAssembly: SIMD backend and Relaxed SIMD

pffft has a dedicated WASM SIMD backend using native `wasm_simd128.h`
intrinsics for both float and double precision.

> **Performance tip:** enable [Relaxed SIMD](https://github.com/WebAssembly/relaxed-simd) for a measured **6-11% throughput gain** on FFT sizes of 1024 and up, on [runtimes that support it](https://caniuse.com/mdn-webassembly_relaxed-simd) (e.g. V8/Chrome, Node.js):
>
> ```sh
> emcmake cmake -DPFFFT_USE_WASM_RELAXED_SIMD=ON ..
> ```
>
> This uses relaxed multiply-add instructions, which the runtime may fuse into a single-rounding FMA, or fall back to a two-rounding multiply followed by add/subtract. It is **off by default** because relaxed SIMD results are not guaranteed bit-identical across engines or hardware (unlike the default WASM SIMD path); enable it only if your application tolerates ULP-level differences between runtimes.

If you build for Emscripten without the bundled CMake config (e.g. a raw
`emcc`/Makefile/Bazel invocation), compile with `-msimd128` to select the WASM
SIMD backend; `PFFFT_ENABLE_WASM` is no longer needed.

## Using pffft in your CMake project

pffft can be included as a dependency in your CMake project via `FetchContent` or `add_subdirectory()`. When used this way, tests, benchmarks, examples, and the standalone `uninstall` helper target are automatically disabled, so pffft won't pollute your build or ctest runner with unnecessary targets or cause target-name collisions.

### FetchContent (CMake 3.11+)

```cmake
include(FetchContent)

FetchContent_Declare(pffft
    GIT_REPOSITORY https://github.com/arancormonk/pffft.git
    GIT_TAG master
)
FetchContent_MakeAvailable(pffft)

target_link_libraries(my_app PRIVATE PFFFT::PFFFT)
```

### add_subdirectory

If you have pffft checked out as a subdirectory or git submodule:

```cmake
add_subdirectory(pffft)

target_link_libraries(my_app PRIVATE PFFFT::PFFFT)
```

### Available targets

| Target | Description |
|---|---|
| `PFFFT::PFFFT` (or `PFFFT`) | FFT library (float and/or double, depending on options) |
| `PFFASTCONV` | Fast convolution library |
| `PFDSP` | DSP utilities (mixer, carrier generation) |

### Build control options

When pffft is the top-level project, these default to `ON`. When included as a subdirectory, they default to `OFF`:

* `PFFFT_BUILD_TESTS` — Build test executables and register CTest tests
* `PFFFT_BUILD_BENCHMARKS` — Build benchmark executables
* `PFFFT_BUILD_EXAMPLES` — Build example programs

Two switches of the FFT pass schedule take `AUTO` (the default), `ON` or
`OFF`, wherever pffft is built; `AUTO` leaves the choice to
`src/pffft_pass_config.h`, which turns both on, except the terminal radix-8
pass on the single-lane scalar build. A parent project's normal variable of
the same name is honoured:

* `PFFFT_TERMINAL_RADIX8` — Replace a plan's radix-2 pass and its last radix-4 pass by one terminal radix-8 pass
* `PFFFT_INPLACE_FIRST_PASS` — Run the first complex pass in place instead of an extra (de)interleave sweep

Builds without CMake may define either macro to 0 or 1. With both off the
passes are those before these switches; see [Pass schedule](#pass-schedule-terminal-radix-8-pass-and-in-place-first-pass).


## History / Origin / Changes:
Origin for this code/fork is Julien Pommier's pffft on bitbucket:
[https://bitbucket.org/jpommier/pffft/](https://bitbucket.org/jpommier/pffft/)

Git history shows following first commits of the major contributors:
* Julien Pommier: November 19, 2011
* Marton Danoczy: September 30, 2015
* Hayati Ayguen: December 22, 2019
* Dario Mambro: March 24, 2020
* George Hilliard: March 15, 2025
* Gunj Joshi: April 1, 2025
* Aleksey Vaneev: December 20, 2025
* Sergi Granell (xerpi): February 26, 2026
* arancormonk: October 12, 2026

There are a few other contributors not listed here.

The main changes on top of the original include:
* improved benchmarking, see [https://github.com/hayguen/pffft_benchmarks](https://github.com/hayguen/pffft_benchmarks)
* double support
* avx(2) support
* c++ headers (wrapper)
* additional API helper functions
* additional library for fast convolution
* const-correct setup arguments for all transform functions
* zero-phase convolution support for linear-phase FIR filters
  (`pffft_zconvert_zp()` / `pffft_zconvolve_zp()`, see `pffft.h`);
  with default scaling the forward transform -> zconvolve_zp ->
  backward pipeline yields the circular convolution at unit gain
* unscaled frequency-domain multiply `pffft_zconvolve()` and `pffft_zconvolve_scale()`
* cmake support
* ctest

The [arancormonk/pffft](https://github.com/arancormonk/pffft) fork of
marton78/pffft, maintained by arancormonk, adds:
* an 8-lane AVX2 implementation of the float transforms on x86-64 (GCC,
  Clang and MSVC, including Windows), selected per setup at runtime on CPUs it
  was measured faster on; see
  [x86: 8-lane AVX2 float transforms](#x86-8-lane-avx2-float-transforms)
* complex transforms that deinterleave and interleave inside their first and
  last FFTPACK passes, on every SIMD backend
* fixes from an accuracy audit against independent long-double reference
  DFTs, full-precision twiddle factors and constants, size limits and
  recoverable setup allocation failures
* the accuracy, allocation, C++ API, mixer and convolution-mode test suites
* fused layout passes for ordered and in-place transforms and vectorized,
  numerically stable PFDSP mixers
* a terminal radix-8 pass in place of the stand-alone radix-2 pass and an
  in-place first complex pass, each one sweep over the data fewer where it
  applies; see [Pass schedule](#pass-schedule-terminal-radix-8-pass-and-in-place-first-pass)
* host-tuned top-level builds with portable CI archives, warnings as errors on
  every platform, CI on Linux, Windows, macOS, ARM64, ARMv7, ppc64le and
  WebAssembly, and CodeQL scanning

The original repository has also continued to evolve independently, receiving
portability fixes and warning cleanups from various contributors. Those changes
have been cherry-picked into this fork where applicable. See the
[Appendix: upstream commit map](#appendix-upstream-commit-map) for a full
commit-by-commit comparison.


## Comparison with other FFTs:
The idea was not to break speed records, but to get a decently fast
fft that is at least 50% as fast as the fastest FFT -- especially on
slowest computers . I'm more focused on getting the best performance
on slow cpus (Atom, Intel Core 1, old Athlons, ARM Cortex-A9...), than
on getting top performance on today fastest cpus.

It can be used in a real-time context as the fft functions do not
perform any memory allocation -- that is why they accept a 'work'
array in their arguments.

It is also a bit focused on performing 1D convolutions, that is why it
provides "unordered" FFTs , and a fourier domain convolution
operation.

Very interesting is [https://www.nayuki.io/page/free-small-fft-in-multiple-languages](https://www.nayuki.io/page/free-small-fft-in-multiple-languages).
It shows how small an FFT can be - including the Bluestein algorithm, but it's everything else than fast.
The whole C++ implementation file is 161 lines, including the Copyright header, see
[https://github.com/nayuki/Nayuki-web-published-code/blob/master/free-small-fft-in-multiple-languages/FftComplex.cpp](https://github.com/nayuki/Nayuki-web-published-code/blob/master/free-small-fft-in-multiple-languages/FftComplex.cpp)

## Dependencies / Required Linux packages

On Debian/Ubuntu Linux following packages should be installed:

```
sudo apt-get install build-essential gcc g++ cmake
```


## Benchmarks and results

#### Quicklink
Find results at [https://github.com/hayguen/pffft_benchmarks](https://github.com/hayguen/pffft_benchmarks).

#### General
My (Hayati Ayguen) first look at FFT-benchmarks was with [benchFFT](http://www.fftw.org/benchfft/)
and especially the results of the benchmarks [results](http://www.fftw.org/speed/),
which demonstrate the performance of the [FFTW](http://www.fftw.org/).
Looking at the benchmarked computer systems from todays view (2021), these are quite outdated.

Having a look into the [benchFFT source code](http://www.fftw.org/benchfft/benchfft-3.1.tar.gz),
the latest source changes, including competitive fft implementations, are dated November 2003.

In 2019, when pffft got my attention at [bitbucket](https://bitbucket.org/jpommier/pffft/src/master/),
there were also some benchmark results.
Unfortunately the results are tables with numbers - without graphical plots.
Without the plots, i could not get an impression. That was, why i started
[https://github.com/hayguen/pffft_benchmarks](https://github.com/hayguen/pffft_benchmarks),
which includes GnuPlot figures.

Today in June 2021, i realized the existence of [https://github.com/FFTW/benchfft](https://github.com/FFTW/benchfft).
This repository is much more up-to-date with a commit in December 2020.
Unfortunately, it looks not so simple to get it run - including the generation of plots.

Is there any website showing benchFFT results of more recent computer systems?

Of course, it's very important, that a benchmark can be compared with a bunch
of different FFT algorithms/implementations.
This requires to have these compiled/built and utilizable.


#### Git submodules for Green-, Kiss- and Pocket-FFT
Sources for [Green-](https://github.com/hayguen/greenffts),
[Kiss-](https://github.com/hayguen/kissfft)
and [Pocket-FFT](https://github.com/hayguen/pocketfft)
can be downloaded directly with the sources of this repository - using git submodules:
```
git clone --recursive https://github.com/arancormonk/pffft.git
```

Important is `--recursive`, that does also fetch the submodules directly.
But you might retrieve the submodules later, too:
```
git submodule update --init
```

#### Fastest Fourier Transform in the West: FFTW
To allow comparison with FFTW [http://www.fftw.org/](http://www.fftw.org/),
cmake option `-DPFFFT_USE_BENCH_FFTW=ON` has to be used with following commands.
The cmake option requires previous setup of following (debian/ubuntu) package:
```
sudo apt-get install libfftw3-dev
```

#### Intel Math Kernel Library: MKL
Intel's MKL [https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)
currently looks even faster than FFTW.

On Ubuntu-Linux it's easy to setup with the package `intel-mkl`.
Similar on Debian: `intel-mkl-full`.

There are special repositories for following Linux distributions:
* Debian/apt: [https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-apt-repo.html](https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-apt-repo.html)
* RedHat/yum: [https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-yum-repo.html](https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-yum-repo.html)
* Gentoo/ebuild: [https://packages.gentoo.org/packages/sci-libs/mkl](https://packages.gentoo.org/packages/sci-libs/mkl)

#### Performing the benchmarks - with CMake
Benchmarks should be prepared by creating a special build folder
```
mkdir build_benches
cd build_benches
cmake ../bench
```

There are several CMake options to parametrize, which fft implementations should be benched.
You can explore all available options with `cmake-gui` or `ccmake`, see [CMake](#cmake).

Some of the options:
* `BENCH_ID`         name the benchmark - used in filename
* `BENCH_ARCH`       target architecture passed to compiler for code optimization
* `PFFFT_USE_BENCH_FFTW`   use (system-installed) FFTW3 in fft benchmark? (default: OFF)
* `PFFFT_USE_BENCH_GREEN`  use Green FFT in fft benchmark? (default: ON)
* `PFFFT_USE_BENCH_KISS`   use KissFFT in fft benchmark? (default: ON)
* `PFFFT_USE_BENCH_POCKET` use PocketFFT in fft benchmark? (default: ON)
* `PFFFT_USE_BENCH_MKL`    use Intel MKL in fft benchmark?  (default: OFF)

These options can be passed to `cmake` at command line, e.g.
```
cmake -DBENCH_ARCH=native -DPFFFT_USE_BENCH_FFTW=ON -DPFFFT_USE_BENCH_MKL=ON ../bench
```

The benchmarks are built and executed with
```
cmake --build .
```

You can also specify to use a different compiler/version with the cmake step, e.g.:

```
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 cmake -DBENCH_ID=gcc9 -DBENCH_ARCH=native -DPFFFT_USE_BENCH_FFTW=ON -DPFFFT_USE_BENCH_MKL=ON ../bench
```

```
CC=/usr/bin/clang-11 CXX=/usr/bin/clang++-11 cmake -DBENCH_ID=clang11 -DBENCH_ARCH=native -DPFFFT_USE_BENCH_FFTW=ON -DPFFFT_USE_BENCH_MKL=ON ../bench
```

For using MSVC/Windows, the cmake command requires/needs the generator and architecture options and to be called from the VS Developer prompt:
```
cmake -G "Visual Studio 16 2019" -A x64 ../bench/
```

see [https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html#visual-studio-generators](https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html#visual-studio-generators)



For running with different compiler version(s):
* copy the result file (.tgz), e.g. `cp *.tgz ../`
* delete the build directory: `rm -rf *`
* then continue with the cmake step


#### Benchmark results and contribution
You might contribute by providing us the results of your computer(s).

The benchmark results are stored in a separate git-repository:
See [https://github.com/hayguen/pffft_benchmarks](https://github.com/hayguen/pffft_benchmarks).

This is to keep this repositories' sources small.


## Appendix: upstream commit map

The table below lists every non-merge commit on
[jpommier/pffft master](https://bitbucket.org/jpommier/pffft/) starting from
the last common ancestor (`16490be`, _"added numbers for the tegra K1"_,
2014-09-03). For each commit the corresponding hash in this fork is given, or
an explanation of why the change was not ported.

| Date | Upstream | Author | Change | Ours | Notes |
|------|----------|--------|--------|------|-------|
| 2016-09-22 | `74d7261` | Julien Pommier | Add 64-bit ARM support | `3673ac0` | |
| 2020-10-15 | `e7de7c2` | Gregor Jasny | Fix 32-bit x86 iOS simulator define | `66ba262` | |
| 2021-05-07 | `82f5a59` | Julien Pommier | Add MKL to benchmark, update results | `e44faf5` | benchmark only |
| 2021-05-07 | `57b818a` | Julien Pommier | Remove old README.txt | -- | already absent |
| 2021-05-11 | `ed78751` | Julien Pommier | Fix compilation when SIMD not available | -- | handled by restructured code |
| 2021-12-01 | `ccd5628` | Mike Hommey | Add missing `#include <altivec.h>` | `b5f1aaa` | cherry-picked |
| 2021-12-02 | `233c266` | Dan Horák | Fix AltiVec vector initialization syntax | `ea9ccdf` | cherry-picked |
| 2021-12-02 | `d3fd82b` | Dan Horák | Fix ppc/ppc64 detection for GCC | `dbaaf40` | cherry-picked |
| 2022-02-15 | `7914df2` | Yair Chuchem | Fix Xcode 13.2.1 warnings | `617dbd9` | cherry-picked (comma-to-semicolons portion; `(void)` part was already done) |
| 2023-03-26 | `045c091` | Yair Chuchem | Fix `-Wshadow` in `pffft_zreorder` | `843d253` | cherry-picked |
| 2023-08-17 | `7ed8b4c` | Julien Pommier | Shorter test time on ARM | -- | test-only; our test infrastructure differs |
| 2023-08-17 | `1adf4b9` | Julien Pommier | Remove `f` suffixes for double-precision safety | `9c0c63c` | cherry-picked |
| 2023-10-11 | `f2455bf` | Nick Dowell | Fix clang `-Wstrict-prototypes` | `f41361d` | already done (void signatures) |
| 2024-03-19 | `17b6d50` | Yair Chuchem | Extract duplicated twiddle code | `2b5b658` | cherry-picked |
| 2024-04-08 | `180c2d6` | Yair Chuchem | Fix `-Wshadow` in `zconvolve_accumulate` | `c8259be` | cherry-picked |
| 2024-11-20 | `c43a62c` | Chris Robinson | Check AltiVec is enabled before using it | `b04ffaa` | cherry-picked |
| 2024-11-25 | `3e10d3b` | Fabian Greffrath | Better SIMD capability checks | `81e6fc3` | cherry-picked (SSE portion; NEON uses CMake option) |
| 2024-11-26 | `8873dd4` | Fabian Greffrath | Define `_USE_MATH_DEFINES` | `1e139d9` | mitigated differently (constants defined directly) |
| 2024-11-29 | `a9fdee9` | Julien Pommier | Add arm64ec support on Windows | `06c8002` | cherry-picked (adapted for split SIMD files) |
| 2024-11-29 | `c0ef461` | Julien Pommier | Replace `long long` with `size_t` | `20a6015` | already uses `uintptr_t` |
| 2024-11-29 | `a8a8013` | Julien Pommier | Enable WASM SIMD support | `ec031bd` | cherry-picked |
| 2024-11-29 | `de2c855` | Julien Pommier | Whitespace cleanup | -- | cosmetic only |
| 2024-11-29 | `6d6a123` | Julien Pommier | `ifac` overflow protection, N validation | `d16aab2` | cherry-picked (library portions) |
| 2024-11-29 | `02fe771` | Julien Pommier | Guard `_USE_MATH_DEFINES` redefinition | -- | not needed (constants defined directly) |
| 2025-02-12 | `0d7449a` | Dan Raviv | Fix MSVC `/fp:strict` C2099 errors | `4d1c78d` | cherry-picked |
| 2025-12-19 | `c306b13` | Julien Pommier | Fix implicit double-to-float conversions | `31be131` | cherry-picked (fftpack.c portion) |
| 2026-01-05 | `0979688` | Julien Pommier | Fix alignment for small `size_t` platforms | `a9786ad` | already uses `uintptr_t` |

For reproducible unfused arithmetic, configure `PFFFT_DISABLE_FMA=ON` and
disable compiler contraction (for GCC/Clang, `-ffp-contract=off`). This option
disables explicit FMA in x86, NEON, AltiVec and relaxed WebAssembly SIMD
backends; it defaults to OFF. Direct source builds may define
`PFFFT_DISABLE_FMA=1` instead. Setup allocation failures return NULL and
release all partial allocations.
