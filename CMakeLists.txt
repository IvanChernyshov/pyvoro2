
cmake_minimum_required(VERSION 3.20)
project(pyvoro2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Voro++ sources (vendored)
set(VORO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/voro++/src")

set(VORO_SOURCES
  "${VORO_SRC_DIR}/c_loops.cc"
  "${VORO_SRC_DIR}/cell.cc"
  "${VORO_SRC_DIR}/common.cc"
  "${VORO_SRC_DIR}/container.cc"
  "${VORO_SRC_DIR}/container_prd.cc"
  "${VORO_SRC_DIR}/pre_container.cc"
  "${VORO_SRC_DIR}/unitcell.cc"
  "${VORO_SRC_DIR}/v_base.cc"
  "${VORO_SRC_DIR}/v_compute.cc"
  "${VORO_SRC_DIR}/wall.cc"
)

pybind11_add_module(_core
  cpp/bindings.cpp
  ${VORO_SOURCES}
)

target_include_directories(_core PRIVATE
  "${VORO_SRC_DIR}"
)

# Some compilers warn about old-style casts inside vendored code; keep warnings reasonable.
if (MSVC)
  target_compile_options(_core PRIVATE /EHsc)
else()
  target_compile_options(_core PRIVATE -O3)
endif()

# Place module under the Python package.
#
# On Unix, Python extension modules are built as shared libraries and
# LIBRARY_OUTPUT_DIRECTORY is sufficient. On Windows, extension modules are
# produced as a DLL-like artifact (".pyd") and CMake treats it as a RUNTIME
# output. For editable installs (pip -e / scikit-build-core metadata_editable),
# we must ensure the extension ends up inside the package directory on all
# platforms.
set(_PYVORO2_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/pyvoro2")
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
  RUNTIME_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${_PYVORO2_OUTDIR}"
)

# Multi-config generators (Visual Studio) use per-config output directories.
# Mirror the same location for all configurations so editable builds can find
# the extension module regardless of the selected config.
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${_cfg}" _cfg_uc)
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
    RUNTIME_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${_cfg_uc} "${_PYVORO2_OUTDIR}"
  )
endforeach()

install(TARGETS _core DESTINATION pyvoro2)
