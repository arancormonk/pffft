# Default architecture for top-level builds: tune for the build host.
#
# provided:
#   - function: pffft_native_arch_default(<C|CXX> <out_var>)
#       sets <out_var> to the TARGET_<LANG>_ARCH default: "native" for GCC/Clang,
#       the highest /arch level the host runs for MSVC-style compilers (which have
#       no /arch:native), or "none" for the compiler's baseline ISA
#   - function: pffft_msvc_host_arch(<out_var>)
#       the /arch level of the build host for MSVC-style compilers, or "none"
#
# The baseline is kept, with a status message saying why, when host tuning would
# be wrong or surprising:
#   - pffft is built as a dependency (add_subdirectory / FetchContent),
#   - cross compiling, Android, iOS or Emscripten,
#   - CMAKE_OSX_ARCHITECTURES names several or a non-host architecture,
#   - the compiler flags of any configuration already select an ISA (-march=,
#     -mcpu=, /arch:), which also covers distribution CFLAGS such as makepkg's
#     -march=x86-64,
#   - the compiler rejects -march=native.
# Packages meant for other machines should configure with --preset release or
# -DTARGET_C_ARCH=none -DTARGET_CXX_ARCH=none.

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(_PFFFT_NATIVE_ARCH_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(pffft_msvc_host_arch out_var)
    if (NOT DEFINED PFFFT_MSVC_HOST_ARCH)
        set(_arch "none")
        try_run(_run_result _compile_result
            "${CMAKE_BINARY_DIR}/pffft_native_arch_probe"
            "${_PFFFT_NATIVE_ARCH_DIR}/native_arch_probe.c"
            RUN_OUTPUT_VARIABLE _output)
        if (_compile_result AND ("${_run_result}" STREQUAL "0")
            AND ("${_output}" MATCHES "PFFFT_HOST_ARCH=([A-Za-z0-9]+)"))
            set(_arch "${CMAKE_MATCH_1}")
        endif()
        set(PFFFT_MSVC_HOST_ARCH "${_arch}" CACHE INTERNAL "/arch level of the build host")
    endif()
    set(${out_var} "${PFFFT_MSVC_HOST_ARCH}" PARENT_SCOPE)
endfunction()

function(pffft_native_arch_default lang out_var)
    set(${out_var} "none" PARENT_SCOPE)
    set(_reason "")
    if (NOT PFFFT_IS_TOPLEVEL)
        set(_reason "built as a dependency")
    elseif (CMAKE_CROSSCOMPILING)
        set(_reason "cross compiling")
    elseif (ANDROID OR IOS OR EMSCRIPTEN)
        set(_reason "${CMAKE_SYSTEM_NAME} target")
    elseif (CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES _n_osx_archs)
        if ( (_n_osx_archs GREATER 1)
             OR NOT ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}") )
            set(_reason "CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' is not the host")
        endif()
    endif()
    if ("${_reason}" STREQUAL "")
        # arch flags apply to every configuration, so check all of them
        string(TOUPPER "${CMAKE_BUILD_TYPE};${CMAKE_CONFIGURATION_TYPES}" _configs)
        set(_flags "${CMAKE_${lang}_FLAGS}")
        foreach (_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL ${_configs})
            set(_flags "${_flags} ${CMAKE_${lang}_FLAGS_${_config}}")
        endforeach()
        if ("${_flags}" MATCHES "(^| )(-march=|-mcpu=|[-/][Aa][Rr][Cc][Hh]:)")
            set(_reason "CMAKE_${lang}_FLAGS already select an architecture")
        endif()
    endif()

    set(_value "none")
    if ("${_reason}" STREQUAL "")
        if ( (CMAKE_${lang}_COMPILER_ID MATCHES "MSVC")
             OR ("${CMAKE_${lang}_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC") )
            if (PFFFT_TARGET_IS_X86)
                pffft_msvc_host_arch(_value)
                if ("${_value}" STREQUAL "none")
                    set(_reason "build host has no AVX")
                endif()
            else()
                set(_reason "no MSVC host tuning for ${CMAKE_SYSTEM_PROCESSOR}")
            endif()
        elseif ( (CMAKE_${lang}_COMPILER_ID STREQUAL "GNU")
                 OR (CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
                 OR (CMAKE_${lang}_COMPILER_ID STREQUAL "AppleClang") )
            if ("${lang}" STREQUAL "C")
                check_c_compiler_flag("-march=native" PFFFT_C_HAS_MARCH_NATIVE)
                set(_supported ${PFFFT_C_HAS_MARCH_NATIVE})
            else()
                check_cxx_compiler_flag("-march=native" PFFFT_CXX_HAS_MARCH_NATIVE)
                set(_supported ${PFFFT_CXX_HAS_MARCH_NATIVE})
            endif()
            if (_supported)
                set(_value "native")
            else()
                set(_reason "compiler rejects -march=native")
            endif()
        else()
            set(_reason "unsupported compiler '${CMAKE_${lang}_COMPILER_ID}'")
        endif()
    endif()

    if ("${_reason}" STREQUAL "")
        message(STATUS "pffft: TARGET_${lang}_ARCH defaults to '${_value}' (tuned for the build host)")
    else()
        message(STATUS "pffft: TARGET_${lang}_ARCH defaults to 'none': ${_reason}")
    endif()
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()
