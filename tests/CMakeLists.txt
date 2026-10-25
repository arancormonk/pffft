if (PFFFT_USE_FFTPACK)
  add_executable(test_fftpack_float ${PFFFT_SOURCE_DIR}/src/fftpack.c ${PFFFT_SOURCE_DIR}/src/fftpack.h)
  target_compile_definitions(test_fftpack_float PRIVATE _USE_MATH_DEFINES TESTING_FFTPACK)
  target_link_libraries(test_fftpack_float ${MATHLIB})

  add_executable(test_fftpack_double ${PFFFT_SOURCE_DIR}/src/fftpack.c ${PFFFT_SOURCE_DIR}/src/fftpack.h)
  target_compile_definitions(test_fftpack_double PRIVATE _USE_MATH_DEFINES FFTPACK_DOUBLE_PRECISION TESTING_FFTPACK)
  target_link_libraries(test_fftpack_double ${MATHLIB})

  # f2c's 1-based indexing ("--wsave;") points one element before the
  # self-test's fixed-size arrays, which GCC then reports as out of bounds
  foreach(t test_fftpack_float test_fftpack_double)
    target_compile_options(${t} PRIVATE $<$<C_COMPILER_ID:GNU>:-Wno-array-bounds>)
  endforeach()

  foreach(prec float double)
    string(TOUPPER "${prec}" PREC)
    add_executable(test_fftpack_accuracy_${prec} test_fftpack_accuracy.c)
    target_link_libraries(test_fftpack_accuracy_${prec} FFTPACK_${PREC} ${ASANLIB} ${MATHLIB})
    if (EMSCRIPTEN)
      # the wsave checks at n = 2 * 3^13 need ~100 MiB (float) to ~200 MiB
      # (double): far more than Emscripten's fixed 16 MiB heap
      target_link_options(test_fftpack_accuracy_${prec} PRIVATE "-sALLOW_MEMORY_GROWTH=1")
    endif()
    add_test(NAME test_fftpack_accuracy_${prec} COMMAND test_fftpack_accuracy_${prec})
  endforeach()
endif()

######################################################

if (PFFFT_USE_TYPE_FLOAT)
  add_executable(test_pf_mixer test_pf_mixer.cpp)
  set_property(TARGET test_pf_mixer PROPERTY CXX_STANDARD 11)
  target_link_libraries(test_pf_mixer PFDSP ${ASANLIB} ${MATHLIB})
  add_test(NAME test_pf_mixer COMMAND test_pf_mixer)

  add_executable(test_pf_mixer_table test_pf_mixer_table.cpp)
  set_property(TARGET test_pf_mixer_table PROPERTY CXX_STANDARD 11)
  target_link_libraries(test_pf_mixer_table PFDSP ${ASANLIB} ${MATHLIB})
  add_test(NAME test_pf_mixer_table COMMAND test_pf_mixer_table)

  add_executable( test_pffft_float  test_pffft.c )
  target_compile_definitions(test_pffft_float PRIVATE _USE_MATH_DEFINES)
  target_compile_definitions(test_pffft_float PRIVATE PFFFT_ENABLE_FLOAT)
  target_link_libraries( test_pffft_float  PFFFT ${ASANLIB} )
endif()

######################################################

if (PFFFT_USE_TYPE_DOUBLE)
  add_executable( test_pffft_double  test_pffft.c )
  target_compile_definitions(test_pffft_double PRIVATE _USE_MATH_DEFINES)
  target_compile_definitions(test_pffft_double PRIVATE PFFFT_ENABLE_DOUBLE)
  target_link_libraries( test_pffft_double  PFFFT ${ASANLIB} )
endif()

######################################################

if (PFFFT_USE_TYPE_FLOAT)
  add_executable( test_pffft_wide  test_pffft_wide.c )
  target_link_libraries( test_pffft_wide  PFFFT ${ASANLIB} ${MATHLIB} )
  target_include_directories( test_pffft_wide PRIVATE ${PFFFT_SOURCE_DIR}/src )
  find_package(Threads)
  if (Threads_FOUND AND NOT WIN32 AND NOT EMSCRIPTEN)
    target_compile_definitions( test_pffft_wide PRIVATE PFFFT_TEST_THREADS )
    target_link_libraries( test_pffft_wide Threads::Threads )
  endif()
  # the default selection test needs the library's CPU policy options
  foreach(_pffft_opt PFFFT_WIDE_UNKNOWN_CPUS PFFFT_WIDE_INFERRED PFFFT_WIDE_ALL_SIZES)
    if (${_pffft_opt})
      target_compile_definitions( test_pffft_wide PRIVATE ${_pffft_opt}=1 )
    endif()
  endforeach()
  if (PFFFT_WIDE_TU_BUILD)
    target_compile_definitions( test_pffft_wide PRIVATE PFFFT_WIDE_TU=1 )
  endif()
  if (MSVC)
    # a map file for check_isa_baseline's audit of the linked code
    set_property(TARGET test_pffft_wide APPEND_STRING PROPERTY LINK_FLAGS " /MAP")
  endif()
  add_test(NAME test_pffft_wide COMMAND test_pffft_wide)
  if (PFFFT_TEST_REQUIRE_WIDE)
    set_tests_properties(test_pffft_wide PROPERTIES ENVIRONMENT "PFFFT_TEST_REQUIRE_WIDE=1")
  endif()
  find_package(Python3 COMPONENTS Interpreter)
  if (Python3_Interpreter_FOUND AND NOT Python3_VERSION VERSION_LESS 3.8)
    # negative controls of the generator of src/pffft_wide_sizes.h
    # (statistics.quantiles needs Python 3.8)
    add_test(NAME wide_size_table_selftest
             COMMAND ${Python3_EXECUTABLE} ${PFFFT_SOURCE_DIR}/benchmarks/perf_analysis/wide_size_table.py --self-test)
  endif()
  # A baseline x86-64 build must keep every instruction beyond SSE2 inside
  # the 8-lane backend, which only runs on CPUs that support it, and that
  # backend must not fault on 16-byte aligned data or mix SSE with dirty
  # AVX state (tests/check_isa_baseline.py). Needs GNU objdump (ELF or
  # PE/COFF); MSVC builds find one on PATH (e.g. MSYS2 binutils).
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
      AND TARGET_C_ARCH MATCHES "^(none|SSE2)$" AND NOT CMAKE_SIZEOF_VOID_P EQUAL 4
      AND Python3_Interpreter_FOUND AND NOT Python3_VERSION VERSION_LESS 3.7)
    # (subprocess.run(capture_output=) needs Python 3.7)
    # GNU objdump: the toolchain's (a cross toolchain's own), else from PATH
    set(_pffft_objdump_default "")
    if (WIN32)
      set(_pffft_od_names x86_64-w64-mingw32-objdump objdump)
    else()
      set(_pffft_od_names objdump x86_64-w64-mingw32-objdump)
    endif()
    foreach(_pffft_od ${CMAKE_OBJDUMP} ${_pffft_od_names})
      unset(_pffft_od_path CACHE)
      find_program(_pffft_od_path NAMES ${_pffft_od})
      if (_pffft_od_path AND NOT _pffft_objdump_default)
        execute_process(COMMAND ${_pffft_od_path} --version OUTPUT_VARIABLE _pffft_od_ver ERROR_QUIET)
        if (_pffft_od_ver MATCHES "^GNU objdump")
          set(_pffft_objdump_default ${_pffft_od_path})
        endif()
      endif()
    endforeach()
    unset(_pffft_od_path CACHE)
    set(PFFFT_ISA_AUDIT_OBJDUMP "${_pffft_objdump_default}" CACHE FILEPATH "GNU objdump for check_isa_baseline")
    set(_pffft_isa_args --objdump ${PFFFT_ISA_AUDIT_OBJDUMP})
    if (CMAKE_C_FLAGS MATCHES "PFFFT_DISABLE_FMA" OR PFFFT_DISABLE_FMA)
      list(APPEND _pffft_isa_args --no-fma)
    endif()
    # the SSE/AVX transition rule is about speed; MSVC /Od compiles scalar
    # code with legacy SSE encodings (a generator expression: Visual Studio
    # picks the configuration at build time)
    list(APPEND _pffft_isa_args $<$<CONFIG:Debug>:--no-transitions>)
    if (PFFFT_WIDE_TU_BUILD)
      list(APPEND _pffft_isa_args --wide-tu)
    endif()
    if (PFFFT_ISA_AUDIT_OBJDUMP)
      # Check the library code itself, not an executable, which may contain
      # the C library's own AVX variants (selected at run time) when linked
      # statically. A static archive built with LTO holds no machine code,
      # so the final executable is checked instead; a shared library always
      # holds the code. MSVC and clang-cl: the objects (MSVC /GL and LLVM
      # bitcode ones hold no machine code and are skipped; all of them may
      # be, the linked image is audited below).
      string(TOUPPER "${CMAKE_BUILD_TYPE}" _pffft_cfg)
      get_target_property(_pffft_type PFFFT TYPE)
      get_target_property(_pffft_ipo PFFFT INTERPROCEDURAL_OPTIMIZATION)
      get_target_property(_pffft_ipo_cfg PFFFT INTERPROCEDURAL_OPTIMIZATION_${_pffft_cfg})
      if (MSVC)
        set(_pffft_isa_files $<TARGET_OBJECTS:PFFFT> --skip-empty --all-il-ok)
      elseif (NOT _pffft_type STREQUAL "SHARED_LIBRARY" AND (_pffft_ipo OR _pffft_ipo_cfg))
        set(_pffft_isa_files $<TARGET_FILE:test_pffft_wide>)
      else()
        set(_pffft_isa_files $<TARGET_FILE:PFFFT>)
      endif()
      add_test(NAME check_isa_baseline
               COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_isa_baseline.py
                       ${_pffft_isa_args} ${_pffft_isa_files})
      # MSVC's linker (or lld-link): the linked code too, by its map file
      # (the image has no symbols), which also covers link-time code
      # generation. Every object of the image except the C runtime's, since
      # then pffft's code may be inlined into the test's. Required: the
      # 8-lane entry points of MSVC's separate translation unit (compiled
      # without /GL) and, unless an object of the library holds no machine
      # code (link-time code generation may inline them into the test), the
      # public functions; a DLL exports them. Where the 8-lane backend must
      # be built, some 8-lane function.
      if (MSVC)
        # objects without a library (also link-time generated ones) or pffft's
        set(_pffft_map_objects "^(pffft:)?[^:<]+$")
        set(_pffft_required "")
        set(_pffft_public pffft_new_setup,pffft_transform)
        if (PFFFT_WIDE_TU_BUILD)
          set(_pffft_required pffft_w8_new_setup,pffft_w8_transform)
        endif()
        if (_pffft_type STREQUAL "SHARED_LIBRARY")
          set(_pffft_image $<TARGET_FILE:PFFFT> --map $<TARGET_FILE_DIR:PFFFT>/pffft.map)
          set(_pffft_required "${_pffft_required},${_pffft_public}")
          set(_pffft_public "")
        else()
          set(_pffft_image $<TARGET_FILE:test_pffft_wide>
              --map $<TARGET_FILE_DIR:test_pffft_wide>/test_pffft_wide.map)
        endif()
        add_test(NAME check_isa_baseline_image
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_isa_baseline.py
                         ${_pffft_isa_args} --map-objects ${_pffft_map_objects}
                         --require "${_pffft_required}" --require-unless-ltcg "${_pffft_public}"
                         --ltcg-objects "$<TARGET_OBJECTS:PFFFT>"
                         $<$<BOOL:${PFFFT_TEST_REQUIRE_WIDE}>:--require-wide> ${_pffft_image})
      endif()
      # negative and positive controls: assembly fixtures in tests/isa_audit
      if (MSVC)
        # COFF fixtures, also linked into DLLs and audited by their maps
        find_program(PFFFT_ISA_AUDIT_CLANG NAMES clang)
        set(_pffft_isa_cc ${PFFFT_ISA_AUDIT_CLANG} --cc-arg --target=x86_64-pc-windows-msvc
            --linker ${CMAKE_LINKER})
      else()
        set(_pffft_isa_cc ${CMAKE_C_COMPILER})
      endif()
      if (NOT MSVC OR PFFFT_ISA_AUDIT_CLANG)
        add_test(NAME check_isa_baseline_selftest
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_isa_baseline.py
                         --self-test --objdump ${PFFFT_ISA_AUDIT_OBJDUMP} --cc ${_pffft_isa_cc})
      endif()
    else()
      message(STATUS "check_isa_baseline: not run, no GNU objdump found")
    endif()
  endif()
endif()

######################################################

foreach(prec float double)
  string(TOUPPER "${prec}" PREC)
  if (PFFFT_USE_TYPE_${PREC})
    add_executable( test_pffft_accuracy_${prec}  test_pffft_accuracy.c )
    target_compile_definitions(test_pffft_accuracy_${prec} PRIVATE PFFFT_ENABLE_${PREC})
    target_link_libraries( test_pffft_accuracy_${prec}  PFFFT ${ASANLIB} ${MATHLIB} )
    target_include_directories( test_pffft_accuracy_${prec} PRIVATE ${PFFFT_SOURCE_DIR}/src )
    if (EMSCRIPTEN)
      # work == NULL puts 2*N scalars of scratch on the stack (128 KiB for
      # complex double at N = 8192): more than Emscripten's default stack
      target_link_options(test_pffft_accuracy_${prec} PRIVATE "-sSTACK_SIZE=1048576")
    endif()
    add_test(NAME test_pffft_accuracy_${prec} COMMAND test_pffft_accuracy_${prec})
  endif()
endforeach()

######################################################

add_executable( test_pffft_cpp_api test_pffft_cpp_api.cpp )
foreach(prec FLOAT DOUBLE)
  if (PFFFT_USE_TYPE_${prec})
    target_compile_definitions(test_pffft_cpp_api PRIVATE PFFFT_ENABLE_${prec})
  endif()
endforeach()
target_link_libraries( test_pffft_cpp_api  PFFFT ${STDCXXLIB} ${ASANLIB} )
if (EMSCRIPTEN)
  # work memory of up to 4096 complex doubles lives on the stack
  target_link_options(test_pffft_cpp_api PRIVATE "-sSTACK_SIZE=1048576")
  # the test catches PFAlloc's std::bad_alloc; Emscripten aborts on throw
  # unless exception support is enabled
  target_compile_options(test_pffft_cpp_api PRIVATE "-fexceptions")
  target_link_options(test_pffft_cpp_api PRIVATE "-fexceptions")
endif()
add_test(NAME test_pffft_cpp_api COMMAND test_pffft_cpp_api)

######################################################

add_executable( test_fft_factors  test_fft_factors.c )
if (PFFFT_USE_TYPE_FLOAT)
  target_compile_definitions(test_fft_factors PRIVATE PFFFT_ENABLE_FLOAT)
endif()
if (PFFFT_USE_TYPE_DOUBLE)
  target_compile_definitions(test_fft_factors PRIVATE PFFFT_ENABLE_DOUBLE)
endif()
target_link_libraries(test_fft_factors PFFFT ${ASANLIB} ${MATHLIB})

######################################################

add_executable( test_pffft_cpp test_pffft.cpp )
target_compile_definitions(test_pffft_cpp PRIVATE _USE_MATH_DEFINES)
if (PFFFT_USE_TYPE_FLOAT)
  target_compile_definitions(test_pffft_cpp PRIVATE PFFFT_ENABLE_FLOAT)
endif()
if (PFFFT_USE_TYPE_DOUBLE)
  target_compile_definitions(test_pffft_cpp PRIVATE PFFFT_ENABLE_DOUBLE)
endif()
target_link_libraries( test_pffft_cpp  PFFFT ${STDCXXLIB} ${ASANLIB} )
if (EMSCRIPTEN)
  # Fft<> keeps its scratch on the stack up to 4096 points (64 KiB for
  # complex double): all of Emscripten's default stack.
  target_link_options(test_pffft_cpp PRIVATE "-sSTACK_SIZE=1048576")
endif()

######################################################

add_executable( test_pffft_cpp_11 test_pffft.cpp )
target_compile_definitions(test_pffft_cpp_11 PRIVATE _USE_MATH_DEFINES)
if (PFFFT_USE_TYPE_FLOAT)
  target_compile_definitions(test_pffft_cpp_11 PRIVATE PFFFT_ENABLE_FLOAT)
endif()
if (PFFFT_USE_TYPE_DOUBLE)
  target_compile_definitions(test_pffft_cpp_11 PRIVATE PFFFT_ENABLE_DOUBLE)
endif()
target_link_libraries( test_pffft_cpp_11  PFFFT ${STDCXXLIB} ${ASANLIB} )
if (EMSCRIPTEN)
  # Fft<> keeps its scratch on the stack up to 4096 points (64 KiB for
  # complex double): all of Emscripten's default stack.
  target_link_options(test_pffft_cpp_11 PRIVATE "-sSTACK_SIZE=1048576")
endif()

set_property(TARGET test_pffft_cpp_11 PROPERTY CXX_STANDARD 11)
set_property(TARGET test_pffft_cpp_11 PROPERTY CXX_STANDARD_REQUIRED ON)

######################################################

if (PFFFT_USE_TYPE_FLOAT)
  add_executable(test_pffastconv   test_pffastconv.c
    ${SIMD_FLOAT_HDRS} ${SIMD_DOUBLE_HDRS}
  )
  target_compile_definitions(test_pffastconv PRIVATE _USE_MATH_DEFINES)
  if (PFFFT_USE_DEBUG_ASAN)
    target_compile_options(test_pffastconv PRIVATE "-fsanitize=address")
  endif()
  target_set_c_arch_flags(test_pffastconv)
  if (NOT PFFFT_USE_SIMD)
    target_compile_definitions(test_pffastconv PRIVATE PFFFT_SIMD_DISABLE=1)
  endif()
  target_include_directories(test_pffastconv PRIVATE ${PFFFT_SOURCE_DIR}/src)
  target_link_libraries( test_pffastconv  PFFASTCONV ${ASANLIB} ${MATHLIB} )
  if (EMSCRIPTEN)
    target_link_options(test_pffastconv PRIVATE "-sALLOW_MEMORY_GROWTH=1")
  endif()

  add_executable(test_pffastconv_modes test_pffastconv_modes.c)
  target_link_libraries(test_pffastconv_modes PFFASTCONV ${ASANLIB} ${MATHLIB})
  add_test(NAME test_pffastconv_modes COMMAND test_pffastconv_modes)
endif()

######################################################

if (PFFFT_USE_TYPE_FLOAT)
  add_test(NAME test_pffft_float
    COMMAND test_pffft_float
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()

if (PFFFT_USE_TYPE_DOUBLE)
  add_test(NAME test_pffft_double
    COMMAND test_pffft_double
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()

# check every SIMD macro of the selected backend against known results
# (a no-op that passes when SIMD_SZ != 4)
foreach(prec float double)
  string(TOUPPER "${prec}" PREC)
  if (PFFFT_USE_TYPE_${PREC})
    add_test(NAME test_pffft_simd_macros_${prec}
      COMMAND test_pffft_${prec} --test-simd
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endforeach()

add_test(NAME test_pffft_cpp
  COMMAND test_pffft_cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME test_pffft_cpp_11
  COMMAND test_pffft_cpp_11
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME test_fft_factors
  COMMAND test_fft_factors
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

if (PFFFT_USE_FFTPACK)
  add_test(NAME test_fftpack_float
    COMMAND test_fftpack_float
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME test_fftpack_double
    COMMAND test_fftpack_double
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()


if (PFFFT_USE_TYPE_FLOAT)

  add_test(NAME test_pfconv_lens_symetric
    COMMAND test_pffastconv "--no-bench" "--quick" "--sym"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME test_pfconv_lens_non_sym
    COMMAND test_pffastconv "--no-bench" "--quick"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME bench_pfconv_symetric
    COMMAND test_pffastconv "--no-len" "--quick" "--sym"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME bench_pfconv_non_sym
    COMMAND test_pffastconv "--no-len" "--quick"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  if (PFFFT_BUILD_BENCHMARKS)
    add_test(NAME bench_pffft_pow2
      COMMAND bench_pffft_float "--max-len" "128" "--quick"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_test(NAME bench_pffft_non2
      COMMAND bench_pffft_float "--non-pow2" "--max-len" "192" "--quick"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()

endif()

# GNU link wrapping sees both allocations even when PFFFT is compiled separately.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT BUILD_SHARED_LIBS)
  add_executable(test_setup_alloc test_setup_alloc.c)
  target_link_libraries(test_setup_alloc PFFFT ${ASANLIB})
  target_include_directories(test_setup_alloc PRIVATE ${PFFFT_SOURCE_DIR}/src)
  target_link_options(test_setup_alloc PRIVATE -Wl,--wrap=malloc -Wl,--wrap=free)
  # The wrappers must be a regular object: under -flto (e.g. distro CFLAGS)
  # the library's LTO-generated malloc/free calls otherwise bypass --wrap.
  target_compile_options(test_setup_alloc PRIVATE -UNDEBUG -fno-lto)
  if(PFFFT_USE_TYPE_FLOAT)
    target_compile_definitions(test_setup_alloc PRIVATE PFFFT_ENABLE_FLOAT)
  endif()
  if(PFFFT_USE_TYPE_DOUBLE)
    target_compile_definitions(test_setup_alloc PRIVATE PFFFT_ENABLE_DOUBLE)
  endif()
  add_test(NAME test_setup_alloc COMMAND test_setup_alloc)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
  add_executable(test_simd_unfused test_simd_unfused.c)
  target_include_directories(test_simd_unfused PRIVATE ${PFFFT_SOURCE_DIR}/src)
  target_compile_definitions(test_simd_unfused PRIVATE PFFFT_DISABLE_FMA=1)
  target_compile_options(test_simd_unfused PRIVATE -ffp-contract=off -UNDEBUG)
  target_set_c_arch_flags(test_simd_unfused)
  if(NOT PFFFT_USE_SIMD)
    target_compile_definitions(test_simd_unfused PRIVATE PFFFT_SIMD_DISABLE=1)
  endif()
  add_test(NAME test_simd_unfused COMMAND test_simd_unfused)
  set_tests_properties(test_simd_unfused PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Keep direct C99 source consumers warning-clean, also with NDEBUG.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
  foreach(mode scalar simd)
    add_library(test_strict_${mode} OBJECT
      ${PFFFT_SOURCE_DIR}/src/pffft.c ${PFFFT_SOURCE_DIR}/src/pffft_common.c)
    target_include_directories(test_strict_${mode} PRIVATE ${PFFFT_SOURCE_DIR}/include/pffft)
    target_compile_definitions(test_strict_${mode} PRIVATE NDEBUG PFFFT_STATIC_DEFINE)
    target_compile_options(test_strict_${mode} PRIVATE -Wall -Wextra -Wpedantic
      -Wstrict-prototypes -Wold-style-definition -Werror -Wno-vla)
    if(mode STREQUAL "scalar")
      target_compile_definitions(test_strict_${mode} PRIVATE PFFFT_SIMD_DISABLE=1)
    endif()
  endforeach()
endif()

# Fresh top-level configures tune for the build host; flags that already pick
# an ISA and dependency builds keep the baseline. Visual Studio configures are
# too slow to repeat here; CI checks the MSVC default separately.
if(PFFFT_IS_TOPLEVEL AND NOT CMAKE_CROSSCOMPILING AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
  # a valid flag that selects the baseline ISA; empty skips that case
  set(_pffft_isa_flag "")
  if(CMAKE_C_COMPILER_ID MATCHES "MSVC" OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    if(PFFFT_TARGET_IS_X86)
      set(_pffft_isa_flag "/arch:AVX")
    endif()
  elseif(PFFFT_TARGET_IS_X86 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 32-bit x86 builds may come from a multilib compiler whose default mode
    # (without the parent's -m32) rejects a 32-bit CPU, so they skip the case
    set(_pffft_isa_flag "-march=x86-64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_pffft_isa_flag "-march=armv8-a")
  elseif(PFFFT_TARGET_IS_ARMV7)
    set(_pffft_isa_flag "-march=armv7-a")
  endif()
  add_test(NAME test_native_default
    COMMAND "${CMAKE_COMMAND}"
      "-DPFFFT_SOURCE_DIR=${PFFFT_SOURCE_DIR}"
      "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/native_default"
      "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      "-DC_COMPILER=${CMAKE_C_COMPILER}"
      "-DC_COMPILER_ARG1=${CMAKE_C_COMPILER_ARG1}"
      "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCXX_COMPILER_ARG1=${CMAKE_CXX_COMPILER_ARG1}"
      "-DISA_FLAG=${_pffft_isa_flag}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_native_default.cmake")
endif()

pffft_copy_runtime_dlls()
