# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#
# cmake -DDEST=<dir> -P pffft_copy_dlls.cmake <file>...: copies the files
# into DEST under a lock on DEST, so that copies of the same DLLs into the
# same directory from parallel post-build steps do not run at once
# (pffft_copy_runtime_dlls() in compiler_warnings.cmake).
file(MAKE_DIRECTORY "${DEST}")
set(_lock "${DEST}/.pffft-dll-copy.lock")
file(LOCK "${_lock}" GUARD PROCESS TIMEOUT 600)
set(_after_script FALSE)
foreach(_k RANGE 1 ${CMAKE_ARGC})
    if(_k EQUAL CMAKE_ARGC)
        break()
    endif()
    set(_arg "${CMAKE_ARGV${_k}}")
    if(_after_script)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_arg}" "${DEST}"
                        RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "copying ${_arg} to ${DEST} failed")
        endif()
    elseif(_arg MATCHES "pffft_copy_dlls\\.cmake$")
        set(_after_script TRUE)
    endif()
endforeach()
file(LOCK "${_lock}" RELEASE)
