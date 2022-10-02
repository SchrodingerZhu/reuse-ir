project(ReuseIR CXX C)
cmake_minimum_required(VERSION 3.20)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
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
add_executable(test src/main.cpp)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(LLVM_LIBS support core irreader)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_subdirectory(include/reuse)
add_subdirectory(src/reuse)

target_link_libraries(test MLIRReuse  ${LLVM_LIBS} ${MLIR_DIALECT_LIBS} ${MLIR_CONVERSION_LIBS})



