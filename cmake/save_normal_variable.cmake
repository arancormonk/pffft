# A parent project may set an option of pffft as a normal variable before
# add_subdirectory(pffft); that is the value in effect. But with
# cmake_minimum_required(VERSION 3.10), policy CMP0126 is OLD here, so
# creating a cache entry removes a normal variable of the same name. Such
# options therefore save a normal variable first and restore it afterwards.
#
# _pffft_save_normal(<entry>)
# A normal variable exists if the name is defined without a cache entry,
# or with a value other than the cache value; one with the cache value
# needs no restoring.
macro(_pffft_save_normal entry)
    get_property(_pffft_type CACHE ${entry} PROPERTY TYPE)
    get_property(_pffft_value CACHE ${entry} PROPERTY VALUE)
    set(_pffft_has_normal_${entry} FALSE)
    if (DEFINED ${entry})
        if (("${_pffft_type}" STREQUAL "") OR NOT ("${${entry}}" STREQUAL "${_pffft_value}"))
            set(_pffft_has_normal_${entry} TRUE)
            set(_pffft_normal_${entry} "${${entry}}")
        endif()
    endif()
endmacro()

# _pffft_restore_normal(<entry>)
macro(_pffft_restore_normal entry)
    if (_pffft_has_normal_${entry})
        set(${entry} "${_pffft_normal_${entry}}")
    endif()
endmacro()
