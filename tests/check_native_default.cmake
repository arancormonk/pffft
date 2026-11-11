# Checks the TARGET_C_ARCH / TARGET_CXX_ARCH defaults of fresh configures:
#   - top-level: tuned for the build host ("native", or the host's /arch level
#     for MSVC-style compilers), unless the compiler or host cannot
#   - CMAKE_C_FLAGS/CMAKE_CXX_FLAGS that already pick an ISA: "none"
#   - built as a dependency through add_subdirectory(): "none"
#
# cmake -DPFFFT_SOURCE_DIR=... -DWORK_DIR=... -DGENERATOR=... [-DMAKE_PROGRAM=...]
#       -DC_COMPILER=... [-DC_COMPILER_ARG1=...] -DCXX_COMPILER=... [-DCXX_COMPILER_ARG1=...]
#       [-DISA_FLAG=<flag selecting the baseline ISA>] -P check_native_default.cmake

# A launcher such as "ccache gcc" keeps the compiler itself in ARG1; pass the
# pair through CC/CXX, which CMake splits again. Plain compilers go through
# -DCMAKE_<LANG>_COMPILER so that paths with spaces stay intact.
set(_env_args --unset=CFLAGS --unset=CXXFLAGS --unset=LDFLAGS --unset=CMAKE_TOOLCHAIN_FILE
    --unset=CMAKE_OSX_ARCHITECTURES)
set(_compiler_args "")
foreach (_lang C CXX)
    string(STRIP "${${_lang}_COMPILER_ARG1}" _arg1)
    if (_arg1)
        set(_env_name CC)
        if (_lang STREQUAL "CXX")
            set(_env_name CXX)
        endif()
        set(_compiler "${${_lang}_COMPILER}")
        if (_compiler MATCHES " ")
            set(_compiler "\"${_compiler}\"")
        endif()
        list(APPEND _env_args "${_env_name}=${_compiler} ${_arg1}")
    else()
        list(APPEND _compiler_args "-DCMAKE_${_lang}_COMPILER=${${_lang}_COMPILER}")
    endif()
endforeach()

set(_common_args
    -G "${GENERATOR}"
    ${_compiler_args}
    -DCMAKE_BUILD_TYPE=Release
    -DPFFFT_BUILD_TESTS=OFF
    -DPFFFT_BUILD_BENCHMARKS=OFF
    -DPFFFT_BUILD_EXAMPLES=OFF)
if (MAKE_PROGRAM)
    list(APPEND _common_args "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()

function(configure_case name source_dir out_log)
    set(_bin "${WORK_DIR}/${name}")
    file(REMOVE_RECURSE "${_bin}")
    file(MAKE_DIRECTORY "${_bin}")
    execute_process(
        # build flags or a toolchain from the environment would select the
        # baseline themselves; -S/-B needs CMake 3.13
        COMMAND "${CMAKE_COMMAND}" -E env ${_env_args}
                "${CMAKE_COMMAND}" ${_common_args} ${ARGN} "${source_dir}"
        WORKING_DIRECTORY "${_bin}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _output)
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR "${name}: configure failed\n${_output}")
    endif()
    set(${out_log} "${_output}" PARENT_SCOPE)
endfunction()

function(read_arch name lang out_var)
    file(STRINGS "${WORK_DIR}/${name}/CMakeCache.txt" _line REGEX "^TARGET_${lang}_ARCH:")
    if (NOT _line)
        message(FATAL_ERROR "${name}: TARGET_${lang}_ARCH missing from the cache")
    endif()
    string(REGEX REPLACE "^[^=]*=" "" _value "${_line}")
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

function(expect_arch name lang expected)
    read_arch(${name} ${lang} _value)
    if (NOT "${_value}" STREQUAL "${expected}")
        message(FATAL_ERROR "${name}: TARGET_${lang}_ARCH is '${_value}', expected '${expected}'")
    endif()
endfunction()

# top-level: tuned for the build host, unless the forwarded compiler command
# itself selects an ISA (CC="gcc -march=x86-64")
configure_case(toplevel "${PFFFT_SOURCE_DIR}" _log)
foreach (_lang C CXX)
    read_arch(toplevel ${_lang} _value)
    set(_allowed "compiler rejects -march=native|build host has no AVX|no MSVC host tuning for ")
    if (" ${${_lang}_COMPILER_ARG1}" MATCHES "[ \t][\"']?(-march=|-mcpu=|[-/][Aa][Rr][Cc][Hh]:)")
        set(_allowed "${_allowed}|the ${_lang} compiler command or flags already select an architecture")
    endif()
    if ("${_value}" STREQUAL "none")
        if (NOT "${_log}" MATCHES "TARGET_${_lang}_ARCH defaults to 'none': (${_allowed})")
            message(FATAL_ERROR "toplevel: TARGET_${_lang}_ARCH is 'none' without a reason\n${_log}")
        endif()
    elseif (NOT "${_value}" MATCHES "^(native|AVX|AVX2|AVX512)$")
        message(FATAL_ERROR "toplevel: TARGET_${_lang}_ARCH is '${_value}', expected the build host")
    endif()
endforeach()

# an ISA chosen in the compiler flags wins, also in a single configuration
# or after a tab
if (ISA_FLAG)
    configure_case(user_flags "${PFFFT_SOURCE_DIR}" _log
        "-DCMAKE_C_FLAGS=${ISA_FLAG}" "-DCMAKE_CXX_FLAGS=${ISA_FLAG}")
    expect_arch(user_flags C "none")
    expect_arch(user_flags CXX "none")
    configure_case(debug_flags "${PFFFT_SOURCE_DIR}" _log
        "-DCMAKE_C_FLAGS_DEBUG=${ISA_FLAG}" "-DCMAKE_CXX_FLAGS_DEBUG=${ISA_FLAG}")
    expect_arch(debug_flags C "none")
    expect_arch(debug_flags CXX "none")
    # single quotes only quote on POSIX command lines
    set(_cxx_quote "'")
    if (CMAKE_HOST_WIN32)
        set(_cxx_quote "\"")
    endif()
    configure_case(quoted_flags "${PFFFT_SOURCE_DIR}" _log
        "-DCMAKE_C_FLAGS=-O2 \"${ISA_FLAG}\""
        "-DCMAKE_CXX_FLAGS=-O2 ${_cxx_quote}${ISA_FLAG}${_cxx_quote}")
    expect_arch(quoted_flags C "none")
    expect_arch(quoted_flags CXX "none")
    configure_case(tab_flags "${PFFFT_SOURCE_DIR}" _log
        "-DCMAKE_C_FLAGS=-O2\t${ISA_FLAG}" "-DCMAKE_CXX_FLAGS=-O2\t${ISA_FLAG}")
    expect_arch(tab_flags C "none")
    expect_arch(tab_flags CXX "none")
endif()

# dependencies keep the compiler's baseline ISA
set(_wrapper "${WORK_DIR}/wrapper_src")
file(MAKE_DIRECTORY "${_wrapper}")
file(WRITE "${_wrapper}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.10)\n"
    "project(pffft_consumer C CXX)\n"
    "add_subdirectory(\"${PFFFT_SOURCE_DIR}\" pffft)\n")
configure_case(subproject "${_wrapper}" _log)
expect_arch(subproject C "none")
expect_arch(subproject CXX "none")

# A parent's normal TARGET_<lang>_ARCH variables are the values in effect,
# also where pffft creates or recreates the cache entries (CMP0126 OLD
# removes a normal variable of the same name then): in a fresh build
# directory, and in one whose cache still holds the choice list of an older
# pffft. The cache value is not what applies, so check what pffft uses.
if (ISA_FLAG)
    string(REGEX REPLACE "^(-march=|/arch:)" "" _parent_arch "${ISA_FLAG}")
    set(_override "${WORK_DIR}/override_src")
    file(MAKE_DIRECTORY "${_override}")
    file(WRITE "${_override}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.10)\n"
        "project(pffft_consumer C CXX)\n"
        "set(TARGET_C_ARCH \"${_parent_arch}\")\n"
        "set(TARGET_CXX_ARCH \"${_parent_arch}\")\n"
        "add_subdirectory(\"${PFFFT_SOURCE_DIR}\" pffft)\n")
    function(expect_parent_arch name log)
        foreach (_msg "C ARCH for target PFFFT set: ${_parent_arch}"
                      "C++ ARCH for target PFDSP set: ${_parent_arch}")
            string(FIND "${log}" "${_msg}" _pos)
            if (_pos LESS 0)
                message(FATAL_ERROR "${name}: the parent's TARGET_<lang>_ARCH="
                                    "${_parent_arch} was not used, expected '${_msg}'\n${log}")
            endif()
        endforeach()
    endfunction()
    configure_case(parent_override "${_override}" _log)
    expect_parent_arch(parent_override "${_log}")

    # as left by an older pffft: another value, and a choice list
    set(_cache "${WORK_DIR}/parent_override/CMakeCache.txt")
    file(READ "${_cache}" _content)
    foreach (_lang C CXX)
        string(REGEX REPLACE "\nTARGET_${_lang}_ARCH:STRING=[^\n]*"
            "\nTARGET_${_lang}_ARCH:STRING=none\nTARGET_${_lang}_ARCH-STRINGS:INTERNAL=none;native"
            _content "${_content}")
    endforeach()
    file(WRITE "${_cache}" "${_content}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${_env_args} "${CMAKE_COMMAND}" .
        WORKING_DIRECTORY "${WORK_DIR}/parent_override"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _log
        ERROR_VARIABLE _log)
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR "parent_override_old_cache: configure failed\n${_log}")
    endif()
    expect_parent_arch(parent_override_old_cache "${_log}")
    # gcc/clang drop the list; MSVC/clang-cl on x86 replace it with the /arch
    # levels. Either way the older pffft's list is gone.
    file(STRINGS "${_cache}" _lists REGEX "^TARGET_(C|CXX)_ARCH-STRINGS:INTERNAL=none;native$")
    if (_lists)
        message(FATAL_ERROR "parent_override_old_cache: the older choice list was kept: ${_lists}")
    endif()
endif()

message(STATUS "native architecture defaults: ok")
