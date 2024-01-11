#ifndef LEAN_TYPE_TAG_ANALYSIS_H
#define LEAN_TYPE_TAG_ANALYSIS_H
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallString.h"
#include <cstddef>
#include <mlir/IR/ValueRange.h>
#include <optional>
#include <string>
#include <utility>

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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
  ChangeResult meet(const TypeTagSemiLattice &rhs);

  /// Print the typed values map
  void print(llvm::raw_ostream &os) const override;

  ::llvm::DenseMap<::mlir::Value, TypeTag> &getTypedValues() {
    return this->typedValues;
  }

private:
  ::llvm::DenseMap<::mlir::Value, TypeTag> typedValues;
};

class TypeTagAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. If this is a
  /// block, then the state is propagated from control-flow predecessors or
  /// callsites. If this is a call operation or region control-flow operation,
  /// then the state after the execution of the operation is set by control-flow
  /// or the callgraph. Otherwise, this function invokes the operation transfer
  /// function.
  LogicalResult visit(ProgramPoint point) override;

private:
  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites.
  void visitBlock(Block *block);
};

} // namespace mlir::dataflow::lean

#endif // LEAN_TYPE_TAG_ANALYSIS_H