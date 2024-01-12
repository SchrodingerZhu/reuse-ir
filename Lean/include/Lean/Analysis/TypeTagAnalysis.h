#ifndef LEAN_TYPE_TAG_ANALYSIS_H
#define LEAN_TYPE_TAG_ANALYSIS_H
#include <cstddef>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <optional>
#include <string>
#include <utility>
namespace mlir::dataflow::lean {

using TypeTag = std::pair<::std::string, std::size_t>;

class TypeTagSemiLattice : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  /// Meet operation
  /// - Idiopotency: Appearantly, not value will be filtered out if meet(x, x);
  /// - Commutativity: meet(x, y) == meet(y, x); the operation is similar to set
  ///   intersection
  /// - Associativity: meet(x, meet(y, z)) == meet(meet(x, y), z); for the same
  ///   reason as above
  static void meet(::llvm::DenseMap<::mlir::Value, TypeTag> &lhs,
                   const ::llvm::DenseMap<::mlir::Value, TypeTag> &rhs);

  /// Print the typed values map
  void print(llvm::raw_ostream &os) const override;

  ::llvm::DenseMap<::mlir::Value, TypeTag> &getTypedValues() {
    return this->typedValues;
  }

  const ::llvm::DenseMap<::mlir::Value, TypeTag> &getTypedValues() const {
    return this->typedValues;
  }

  ChangeResult setTypedValue(::llvm::DenseMap<::mlir::Value, TypeTag> map) {
    if (this->typedValues == map)
      return ChangeResult::NoChange;
    this->typedValues = std::move(map);
    return ChangeResult::Change;
  }

private:
  ::llvm::DenseMap<::mlir::Value, TypeTag> typedValues{};
};

class TypeTagAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  void visitBlock(Block *block);
  void initializeRecursively(Operation *top);
  std::optional<std::pair<Value, TypeTag>> getGenTypeTag(Block *block);
};

struct RunTypeTagAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunTypeTagAnalysis)

  RunTypeTagAnalysis(Operation *op);

  const TypeTagSemiLattice *getKnownTypeTags(Block *block);

private:
  /// Stores the result of the liveness analysis that was run.
  DataFlowSolver solver;
};

} // namespace mlir::dataflow::lean

#endif // LEAN_TYPE_TAG_ANALYSIS_H