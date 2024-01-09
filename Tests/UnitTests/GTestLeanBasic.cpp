#include <gtest/gtest.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>

#include "Lean/LeanOpsDialect.h"
#include "Refcnt/RefcntOpsDialect.h"

namespace mlir {
static std::unique_ptr<MLIRContext> defaultContext() {
  auto context = std::make_unique<MLIRContext>();
  auto registry = DialectRegistry{};
  registerAllDialects(registry);
  registry.insert<refcnt::RefcntDialect, lean::LeanDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  return context;
}
TEST(GTestLeanBasic, DialectRegistration) {
  auto context = defaultContext();
  EXPECT_TRUE(context->getLoadedDialect<refcnt::RefcntDialect>() != nullptr);
  EXPECT_TRUE(context->getLoadedDialect<lean::LeanDialect>() != nullptr);
}
TEST(GTestLeanBasic, ParseBasicSource) {
  const char *source = R"(
  module {
    func.func @test(%list: !refcnt.rc<!lean.obj<2, 0>>) -> !refcnt.rc<i64> {
      %0 = refcnt.new : !refcnt.rc<i64>
      refcnt.dec %list : !refcnt.rc<!lean.obj<2, 0>>
      return %0 : !refcnt.rc<i64>
    }
  }
  )";
  auto context = defaultContext();
  auto module = parseSourceString(source, context.get());
  ScopedDiagnosticHandler error_handler(context.get());
  ASSERT_TRUE(module.get() != nullptr);
  module.get()->dump();
}
} // namespace mlir