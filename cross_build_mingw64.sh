#!/bin/bash

# Debian/Ubuntu prerequisites: gcc-mingw-w64 and g++-mingw-w64.
if [ -z "$1" ]; then
  echo "usage: $0 <build suffix> [cmake options...]" >&2
  exit 1
fi

suffix=$1
shift
build_dir="build_w64_$suffix"
# Nested CMake compiler probes must also use the MinGW toolchain.
unset CMAKE_TOOLCHAIN_FILE
cmake -S . -B "$build_dir" \
  -DCMAKE_TOOLCHAIN_FILE=mingw-w64-x64_64.cmake \
  "-DCMAKE_INSTALL_PREFIX=pffft_bin-w64_$suffix" \
  "$@" && cmake --build "$build_dir"
