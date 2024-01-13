cmake_minimum_required(VERSION 3.20)
project(ReuseAnalysis CXX C)

option(REUSE_ANALYSIS_ENABLE_TESTS "Build Refcnt tests" ON)

# use lld if clang is used
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using lld as linker")
  add_link_options("-fuse-ld=lld")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(${MLIR_INCLUDE_DIRS})
separate_arguments(MLIR_DEFINITIONS_LIST NATIVE_COMMAND ${MLIR_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS_LIST})

message(STATUS "include dirs for llvm and mlir: ${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}")

# Now build our tools


# Find the libraries that correspond to the LLVM components
# that we wish to use

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

if (REUSE_ANALYSIS_ENABLE_TESTS)
  enable_testing()  
  add_subdirectory(test)
endif()