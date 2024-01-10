#include <gtest/gtest.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
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

#define SOURCE_PARSE_TEST(SUITE, NAME, SOURCE)                                 \
  TEST(SUITE, NAME) {                                                          \
    auto context = defaultContext();                                           \
    auto module = parseSourceString(SOURCE, context.get());                    \
    ScopedDiagnosticHandler error_handler(context.get());                      \
    ASSERT_TRUE(module.get() != nullptr);                                      \
    module.get()->dump();                                                      \
  }

SOURCE_PARSE_TEST(GTestLeanBasic, ParseBasic, R"(
  module {
    func.func @test(%list: !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj> {
      %0 = refcnt.new { ctor = "List.Nil" , tag = 0 } : !refcnt.rc<!lean.obj> 
      refcnt.dec %list : !refcnt.rc<!lean.obj>
      return %0 : !refcnt.rc<!lean.obj>
    }
  }
  )")

SOURCE_PARSE_TEST(GTestLeanBasic, ProjectionAndApp, R"(
  module {
    func.func @test(%f: !refcnt.rc<!lean.obj>, %x : !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj> {
      %0 = lean.proj %x, 0 : index
      %1 = lean.proj %x, 1 : index
      refcnt.inc %0 : !refcnt.rc<!lean.obj>
      refcnt.inc %1 : !refcnt.rc<!lean.obj>
      %2 = lean.app %f : !refcnt.rc<!lean.obj>, (%0, %1 : !refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>), !refcnt.rc<!lean.obj>
      return %2 : !refcnt.rc<!lean.obj>
    }
  }
  )")

} // namespace mlir