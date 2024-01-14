#include "Lean/Pass/TestReusabilityAnalysis.h"
#include "Lean/Analysis/ReusabilityAnalysis.h"
#include "Refcnt/IR/RefcntOps.h"
namespace mlir::lean {
void TestReusabilityAnalysisPass::runOnOperation() {
  Operation *op = getOperation();
  ReusabilityAnalysis analysis(op);
  printAnalysisResults(llvm::errs(), op, analysis);
}
void TestReusabilityAnalysisPass::printAnalysisResults(
    raw_ostream &os, Operation *op, const ReusabilityAnalysis &analysis) {
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
            auto analysisResult = analysis.getReusabilityTable().find(res);
            if (analysisResult != analysis.getReusabilityTable().end()) {
              os << "  ";
              analysisResult->first.printAsOperand(os, asmState);
              os << " = {";
              for (auto &reusable : analysisResult->second) {
                reusable->dump(os, asmState);
                os << " ";
              }
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