#include "Lean/Pass/TestReusabilityAnalysis.h"
#include "Lean/Analysis/ReusabilityAnalysis.h"
#include "Lean/Analysis/TypeTagAnalysis.h"
#include "Refcnt/IR/RefcntOps.h"
#include "mlir/Analysis/DataFlowFramework.h"
namespace mlir::lean {
void TestReusabilityAnalysisPass::runOnOperation() {
  Operation *op = getOperation();
  DataFlowSolver solver;
  solver.load<TypeTagAnalysis>();
  solver.load<ReusabilityAnalysis>();
  (void)solver.initializeAndRun(op);
  printAnalysisResults(llvm::errs(), op, solver);
}
void TestReusabilityAnalysisPass::printAnalysisResults(raw_ostream &os,
                                                       Operation *op,
                                                       DataFlowSolver &solver) {
  op->walk([&](Operation *op) {
    auto asmState = AsmState(op);
    auto tag = op->getAttrOfType<StringAttr>("tag");
    if (!tag)
      return;
    os << tag.getValue() << ":\n";
    for (Region &region : op->getRegions()) {
      os << " region #" << region.getRegionNumber() << "\n";
      for (Block &block : region) {
        os << "  ";
        block.printAsOperand(os);
        os << ": ";
        for (auto &op : block.getOperations()) {
          if (auto newOp = dyn_cast<refcnt::NewOp>(op)) {
            auto res = newOp.getResult();
            auto analysisResult =
                solver.lookupState<refcnt::ReusabibilityLookupTable>(newOp);
            if (analysisResult) {
              os << "  ";
              res.printAsOperand(os, asmState);
              os << " = {";
              llvm::interleaveComma(*analysisResult, os, [&](auto &reusable) {
                reusable->dump(os, asmState);
              });
              os << "}\n";
            }
          }
        }
        os << "\n";
      }
    }
  });
}
} // namespace mlir::lean