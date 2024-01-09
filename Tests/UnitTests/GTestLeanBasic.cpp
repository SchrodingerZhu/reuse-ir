#include "gtest/gtest.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>

TEST(GTestLeanBasic, Test1) {
  auto context = mlir::MLIRContext{};
  EXPECT_EQ(1, 1);
}