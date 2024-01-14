#ifndef REFCNT_ANALYSIS_REUSE_ANALYSIS_H
#define REFCNT_ANALYSIS_REUSE_ANALYSIS_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Value.h>

#include "Refcnt/Utilities/ADT.h"

namespace mlir::refcnt {
class Reusable {
public:
  /// Return reusibility estimation of current value.
  /// Negative value means that current value is not reusable.
  virtual size_t reusability() const = 0;
  virtual void dump(llvm::raw_ostream &os, AsmState &) const;
  virtual ~Reusable() = default;
  const Value &getValue() const { return value; }
  Value &getValue() { return value; }
  TypeID getKind() const { return kind; }
  Reusable(TypeID kind, Value value) : kind(kind), value(std::move(value)) {}

private:
  TypeID kind;
  Value value;
};

struct ReusabilityComparator {
  bool operator()(const std::unique_ptr<Reusable> &lhs,
                  const std::unique_ptr<Reusable> &rhs) const {
    return lhs->reusability() > rhs->reusability();
  }
};

class ReusabibilityLookupTable
    : public llvm::DenseMap<Value, refcnt::OrderedSet<std::unique_ptr<Reusable>,
                                                      ReusabilityComparator>> {
public:
  Reusable *getMostPreferredReusable(Value value) const;
};
} // namespace mlir::refcnt
#endif // REFCNT_ANALYSIS_REUSE_ANALYSIS_H