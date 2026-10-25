
# GCC-style options; clang-cl (MSVC is set) gets /W4 from
# pffft_apply_project_warnings() instead and rejects -pedantic
function(target_activate_cxx_compiler_warnings target)
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>)
        target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>)
    endif()
endfunction()

function(target_activate_c_compiler_warnings target)
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE $<$<C_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>)
        target_compile_options(${target} PRIVATE $<$<C_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>)
    endif()
endfunction()


# Warnings (as errors with PFFFT_WARNINGS_AS_ERRORS) for every buildsystem
# target defined directly in the given source directories -- the library,
# tests, benchmarks and examples, but not vendored third-party projects
# pulled in with add_subdirectory().
function(pffft_apply_project_warnings)
    if(MSVC)
        # narrowing double->float conversions of constants are deliberate
        set(_pffft_warn /W4 /wd4244 /wd4305)
        set(_pffft_werror /WX)
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set(_pffft_warn -Wall -Wextra -pedantic)
        set(_pffft_werror -Werror)
    else()
        return()
    endif()
    if(PFFFT_WARNINGS_AS_ERRORS)
        list(APPEND _pffft_warn ${_pffft_werror})
    endif()
    foreach(_dir ${ARGN})
        get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(_target ${_targets})
            get_target_property(_type ${_target} TYPE)
            if(NOT _type MATCHES "INTERFACE_LIBRARY|UTILITY")
                target_compile_options(${_target} PRIVATE ${_pffft_warn})
                if(MSVC)
                    # fopen(), sprintf() etc. are standard C, not deprecated
                    target_compile_definitions(${_target} PRIVATE _CRT_SECURE_NO_WARNINGS)
                endif()
            endif()
        endforeach()
    endforeach()
endfunction()


# An output directory as CMake uses it: a relative one is relative to the
# binary directory of the target's directory, also when a generator
# expression yields it (resolved by $<PATH:...>, CMake 3.24; with an older
# CMake, fallback is set instead).
set(PFFFT_COPY_DLLS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/pffft_copy_dlls.cmake")

function(_pffft_resolve_output_dir out dir fallback)
    if(dir MATCHES "\\$<")
        if(CMAKE_VERSION VERSION_LESS 3.24)
            set(${fallback} TRUE PARENT_SCOPE)
            set(${out} "${dir}" PARENT_SCOPE)
        else()
            set(${out} "$<IF:$<PATH:IS_ABSOLUTE,${dir}>,${dir},${CMAKE_CURRENT_BINARY_DIR}/${dir}>" PARENT_SCOPE)
        endif()
    elseif(IS_ABSOLUTE "${dir}")
        set(${out} "${dir}" PARENT_SCOPE)
    else()
        set(${out} "${CMAKE_CURRENT_BINARY_DIR}/${dir}" PARENT_SCOPE)
    endif()
endfunction()

# Windows has no rpath: a shared build copies the project's DLLs next to the
# executables defined in the calling directory (tests, benchmarks,
# examples), so that ctest and direct runs find them. One copy step per
# destination, shared by every executable writing there (also from other
# directories): executables linking in parallel must not copy the same DLL
# into the same place at the same time.
function(pffft_copy_runtime_dlls)
    if(NOT (WIN32 AND BUILD_SHARED_LIBS))
        return()
    endif()
    set(_files "")
    set(_libs "")
    foreach(_lib PFFFT PFDSP PFFASTCONV)
        if(TARGET ${_lib})
            get_target_property(_lib_type ${_lib} TYPE)
            if(_lib_type STREQUAL "SHARED_LIBRARY")
                list(APPEND _libs ${_lib})
                list(APPEND _files $<TARGET_FILE:${_lib}>)
            endif()
        endif()
    endforeach()
    if(NOT _libs)
        return()
    endif()
    get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_multi)
        set(_configs ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(_configs ${CMAKE_BUILD_TYPE})
    endif()
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target ${_targets})
        get_target_property(_type ${_target} TYPE)
        if(NOT _type STREQUAL "EXECUTABLE")
            continue()
        endif()
        # the target's output directory, named here (as $<TARGET_FILE_DIR> it
        # would make the copy depend on the executable): the directory of its
        # configuration if set, else the generic one, where multi-configuration
        # generators append the configuration unless it holds a generator
        # expression
        set(_fallback FALSE)
        get_target_property(_generic ${_target} RUNTIME_OUTPUT_DIRECTORY)
        if(NOT _generic)
            set(_generic "${CMAKE_CURRENT_BINARY_DIR}")
        else()
            _pffft_resolve_output_dir(_generic "${_generic}" _fallback)
        endif()
        if(_multi AND NOT _generic MATCHES "\\$<")
            set(_generic "${_generic}/$<CONFIG>")
        endif()
        set(_dir "")
        set(_per_config FALSE)
        foreach(_cfg ${_configs})
            string(TOUPPER "${_cfg}" _ucfg)
            get_target_property(_d ${_target} RUNTIME_OUTPUT_DIRECTORY_${_ucfg})
            if(_d)
                set(_per_config TRUE)
                _pffft_resolve_output_dir(_d "${_d}" _fallback)
            else()
                set(_d "${_generic}")
            endif()
            string(APPEND _dir "$<$<CONFIG:${_cfg}>:${_d}>")
        endforeach()
        if(NOT _per_config)
            set(_dir "${_generic}")
        endif()
        if(_fallback)
            # a generator expression this CMake cannot resolve: copy after
            # linking, into the executable's own directory, under a lock on
            # it (several executables may share it)
            add_dependencies(${_target} ${_libs})
            add_custom_command(TARGET ${_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -DDEST=$<TARGET_FILE_DIR:${_target}>
                        -P ${PFFFT_COPY_DLLS_SCRIPT} ${_files}
                VERBATIM)
            continue()
        endif()
        # one copy target per destination expression, for the whole build;
        # different expressions may still name the same directory (bin/$<CONFIG>
        # and bin/Release), so the copy targets run one after another
        string(MD5 _id "${_dir}")
        string(SUBSTRING "${_id}" 0 12 _id)
        set(_copy pffft_runtime_dlls_${_id})
        if(NOT TARGET ${_copy})
            # the same locked copy as the fallback below: copies into one
            # directory exclude each other whichever way they are made
            add_custom_target(${_copy}
                COMMAND ${CMAKE_COMMAND} "-DDEST=${_dir}" -P ${PFFFT_COPY_DLLS_SCRIPT} ${_files}
                DEPENDS ${_libs}
                VERBATIM)
            get_property(_last GLOBAL PROPERTY PFFFT_LAST_DLL_COPY)
            if(_last)
                add_dependencies(${_copy} ${_last})
            endif()
            set_property(GLOBAL PROPERTY PFFFT_LAST_DLL_COPY ${_copy})
        endif()
        add_dependencies(${_target} ${_copy})
    endforeach()
endfunction()
