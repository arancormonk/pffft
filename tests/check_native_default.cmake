# Checks the TARGET_C_ARCH / TARGET_CXX_ARCH defaults of fresh configures:
#   - top-level: tuned for the build host ("native", or the host's /arch level
#     for MSVC-style compilers), unless the compiler or host cannot
#   - CMAKE_C_FLAGS/CMAKE_CXX_FLAGS that already pick an ISA: "none"
#   - built as a dependency through add_subdirectory(): "none"
#
# cmake -DPFFFT_SOURCE_DIR=... -DWORK_DIR=... -DGENERATOR=... [-DMAKE_PROGRAM=...]
#       -DC_COMPILER=... -DCXX_COMPILER=... -DISA_FLAG=<flag selecting an ISA>
#       -P check_native_default.cmake

set(_common_args
    -G "${GENERATOR}"
    "-DCMAKE_C_COMPILER=${C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
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
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${source_dir}" -B "${_bin}" ${_common_args} ${ARGN}
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

# top-level: tuned for the build host
configure_case(toplevel "${PFFFT_SOURCE_DIR}" _log)
foreach (_lang C CXX)
    read_arch(toplevel ${_lang} _value)
    if ("${_value}" STREQUAL "none")
        if (NOT "${_log}" MATCHES "TARGET_${_lang}_ARCH defaults to 'none': (compiler rejects -march=native|build host has no AVX)")
            message(FATAL_ERROR "toplevel: TARGET_${_lang}_ARCH is 'none' without a reason\n${_log}")
        endif()
    elseif (NOT "${_value}" MATCHES "^(native|AVX|AVX2|AVX512)$")
        message(FATAL_ERROR "toplevel: TARGET_${_lang}_ARCH is '${_value}', expected the build host")
    endif()
endforeach()

# an ISA chosen in the compiler flags wins
configure_case(user_flags "${PFFFT_SOURCE_DIR}" _log
    "-DCMAKE_C_FLAGS=${ISA_FLAG}" "-DCMAKE_CXX_FLAGS=${ISA_FLAG}")
expect_arch(user_flags C "none")
expect_arch(user_flags CXX "none")

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

message(STATUS "native architecture defaults: ok")
