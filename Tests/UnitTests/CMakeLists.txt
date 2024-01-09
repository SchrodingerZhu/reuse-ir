find_package(GTest REQUIRED CONFIG)
include(GoogleTest)

add_executable(GTestLeanBasic GTestLeanBasic.cpp)

llvm_map_components_to_libnames(LLVM_LIBS support)
get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

target_link_libraries(GTestLeanBasic PRIVATE MLIRLean MLIRRefcnt ${LLVM_LIBS} ${MLIR_LIBS} GTest::gtest GTest::gtest_main)
gtest_discover_tests(GTestLeanBasic)
