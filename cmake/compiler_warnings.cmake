
function(target_activate_cxx_compiler_warnings target)
    target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>)
    target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>)
endfunction()

function(target_activate_c_compiler_warnings target)
    target_compile_options(${target} PRIVATE $<$<C_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>)
    target_compile_options(${target} PRIVATE $<$<C_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>)
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
