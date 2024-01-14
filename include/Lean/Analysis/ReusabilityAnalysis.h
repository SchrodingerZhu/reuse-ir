#ifndef LEAN_ANALYSIS_REUSABILITY_ANALYSIS_H
#define LEAN_ANALYSIS_REUSABILITY_ANALYSIS_H
#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Value.h>

#include "Refcnt/Analysis/ReuseAnalysis.h"
#include "Refcnt/IR/RefcntOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::lean {

inline constexpr size_t OBJECT_SIMILARITY_SCORE = 5;
inline constexpr size_t SCALAR_SIMILARITY_SCORE = 1;

class ReusableObject : public refcnt::Reusable {
public:
  /// Type casting support
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReusableObject)
  static bool classof(const refcnt::Reusable *obj) {
    return obj->getKind() == resolveTypeID();
  }
  size_t reusibility() const override {
    return reusableObjs.size() * OBJECT_SIMILARITY_SCORE +
           reusableScalars.size() * SCALAR_SIMILARITY_SCORE;
  }
  ReusableObject(Value value, llvm::SmallVector<size_t, 4> reusableObjs,
                 llvm::SmallVector<size_t, 4> reusableScalars)
      : refcnt::Reusable(resolveTypeID(), value),
        reusableObjs(std::move(reusableObjs)),
        reusableScalars(std::move(reusableScalars)) {}

  llvm::SmallVector<size_t, 4> &getReusableObjs() { return reusableObjs; }
  llvm::SmallVector<size_t, 4> &getReusableScalars() { return reusableScalars; }
  const llvm::SmallVector<size_t, 4> &getReusableObjs() const {
    return reusableObjs;
  }
  const llvm::SmallVector<size_t, 4> &getReusableScalars() const {
    return reusableScalars;
  }

private:
  llvm::SmallVector<size_t, 4> reusableObjs;
  llvm::SmallVector<size_t, 4> reusableScalars;
};

class ReusabilityAnalysis {
public:
  ReusabilityAnalysis(Operation *op);
  refcnt::ReusabibilityLookupTable &getReusabilityTable() {
    return reusabilityTable;
  }
  const refcnt::ReusabibilityLookupTable &getReusabilityTable() const {
    return reusabilityTable;
  }

private:
  refcnt::ReusabibilityLookupTable reusabilityTable;
  DataFlowSolver solver;
  void visitNewOp(refcnt::NewOp newOp);
  void analyzeRecursively(Operation *op);
};

} // namespace mlir::lean
#endif // LEAN_ANALYSIS_REUSABILITY_ANALYSIS_H