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
} // namespace mlir::refcnt