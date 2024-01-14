#include "Refcnt/Analysis/ReuseAnalysis.h"
namespace mlir::refcnt {
Reusable *
ReusabibilityLookupTable::getMostPreferredReusable(Value value) const {
  auto it = find(value);
  if (it == end() || it->second.empty())
    return nullptr;
  return it->second.begin()->get();
}
void Reusable::dump(llvm::raw_ostream &os, AsmState &st) const {
  os << "[";
  value.printAsOperand(os, st);
  os << ", score: " << reusability() << "]";
};
void ReuseTokenCollection::print(llvm::raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(*this, os, [&](auto token) {
    Value val = token->getReusedValue();
    auto op = val.getDefiningOp();
    auto asmState = AsmState(op);
    val.printAsOperand(os, asmState);
    if (token->getKind() == ReuseToken::Kind::Conditional) {
      os << "[conditional]";
    } else {
      os << "[direct]";
    }
  });
  os << "}";
}
double ConditonalReuseToken::getReuseProbability(DataFlowSolver &solver) const {
  auto blk = getJoinBlock();
  size_t total = 0;
  size_t valid = 0;
  for (auto pred : blk->getPredecessors()) {
    total += 1;
    if (pred->mightHaveTerminator()) {
      auto term = pred->getTerminator();
      auto collection = solver.lookupState<ReuseTokenCollection>(term);
      if (collection && collection->contains(getSource())) {
        valid += 1;
      }
    }
  }
  // TODO: also consider branch probability.
  return static_cast<double>(valid) / static_cast<double>(total);
}
} // namespace mlir::refcnt