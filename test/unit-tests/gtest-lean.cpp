#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>

#include "Lean/Analysis/TypeTagAnalysis.h"
#include "Lean/IR/LeanOpsDialect.h"
#include "Refcnt/IR/RefcntOpsDialect.h"

namespace mlir::lean {
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
      refcnt.dec %x : !refcnt.rc<!lean.obj>
      %2 = lean.app %f (%0, %1)
      return %2 : !refcnt.rc<!lean.obj>
    }
  }
  )")

SOURCE_PARSE_TEST(GTestLeanBasic, Tag, R"(
  module {
    func.func @test(%obj: !refcnt.rc<!lean.obj>) -> i16 {
      %0 = lean.tag %obj
      refcnt.dec %obj : !refcnt.rc<!lean.obj>
      return %0 : i16
    }
  }
  )")

SOURCE_PARSE_TEST(GTestLeanBasic, ScalarProjection, R"(
  module {
    func.func @test(%obj: !refcnt.rc<!lean.obj>) -> f64 {
      %0 = lean.sproj %obj, 0 : index, f64
      refcnt.dec %obj : !refcnt.rc<!lean.obj>
      return %0 : f64
    }
  }
  )")

SOURCE_PARSE_TEST(GTestLeanBasic, ScalarSet, R"(
  module {
    func.func @test(%x : f64) -> !refcnt.rc<!lean.obj> {
      %0 = refcnt.new { ctor = "LeanFloat" , tag = 0 } : !refcnt.rc<!lean.obj>
      lean.sset %0, 0 : index, %x : f64
      return %0 : !refcnt.rc<!lean.obj>
    }
  }
  )")

const std::string_view REVERSE_MAP = R"(
  module {
    func.func @rev_map(%f: !refcnt.rc<!lean.obj>, %xs : !refcnt.rc<!lean.obj>, %acc : !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj> {
      cf.br ^start(%xs, %acc: !refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>)
    ^start(%list: !refcnt.rc<!lean.obj>, %updated : !refcnt.rc<!lean.obj>):
      %tag = lean.tag %list
      cf.switch %tag : i16, [
        default: ^bb_default,
        0: ^bb_nil,
        1: ^bb_cons
      ] { lean.type = "List" }
    ^bb_default:
      llvm.unreachable
    ^bb_nil:
      refcnt.dec %f : !refcnt.rc<!lean.obj>
      refcnt.dec %list : !refcnt.rc<!lean.obj>
      return %updated : !refcnt.rc<!lean.obj>
    ^bb_cons:
      %hd = lean.proj %list, 0 : index
      %tl = lean.proj %list, 1 : index
      refcnt.inc %hd : !refcnt.rc<!lean.obj>
      refcnt.inc %tl : !refcnt.rc<!lean.obj>
      refcnt.dec %list : !refcnt.rc<!lean.obj>
      refcnt.inc %f : !refcnt.rc<!lean.obj>
      %y = lean.app %f (%hd)
      %cons = refcnt.new { lean.type = "List" , lean.tag = 1 } : !refcnt.rc<!lean.obj>
      lean.set %cons, 0 : index, %y
      lean.set %cons, 1 : index, %updated
      cf.br ^start(%tl, %cons: !refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>)
    }
  }
  )";

SOURCE_PARSE_TEST(GTestLeanBasic, ReverseMap, REVERSE_MAP)

TEST(GTestLeanBasic, ReverseTypeTagAnalysis) {
  auto context = defaultContext();
  auto module = parseSourceString(REVERSE_MAP, context.get());
  ScopedDiagnosticHandler error_handler(context.get());
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<lean::TypeTagAnalysis>();
  auto func = &*module.get()->getRegions().front().getOps().begin();
  ASSERT_TRUE(solver.initializeAndRun(func).succeeded());
  auto funcOp = dyn_cast<func::FuncOp>(func);
  auto &region = funcOp.getBody();
  for (Block &block : region) {
    auto executable = solver.lookupState<dataflow::Executable>(&block);
    auto lattice = solver.lookupState<lean::TypeTagSemiLattice>(&block);
    block.dump();
    if (lattice) {
      llvm::outs() << "  - executable: " << executable->isLive() << "\n"
                   << "  - type-tag: " << *lattice << "\n";
    }
  }
}

} // namespace mlir::lean