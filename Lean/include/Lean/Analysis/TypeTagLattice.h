#ifndef LEAN_ANALYSIS_TYPE_TAG_LATTICE_H
#define LEAN_ANALYSIS_TYPE_TAG_LATTICE_H
#include <cstddef>
#include <optional>
#include <string>

#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
namespace mlir::lean {
using namespace dataflow;
struct TypeTag {
  /// Fully qualified type name (type params are erased).
  ::std::string type;
  /// Constructor Tag.
  ::std::size_t tag;

  bool operator==(const TypeTag &other) const {
    return type == other.type && tag == other.tag;
  }
};
class TypeTagLatticeValue {
public:
  explicit TypeTagLatticeValue() = default;
  explicit TypeTagLatticeValue(TypeTag given) : typeTag(::std::move(given)) {}

  TypeTag getTypeTag() const {
    assert(typeTag.has_value() && "TypeTagLatticeValue is not initialized");
    return typeTag.value();
  }

  bool operator==(const TypeTagLatticeValue &other) const {
    return typeTag == other.typeTag;
  }

  bool isUnknown() const { return !typeTag.has_value(); }

  TypeTagLatticeValue join(const TypeTagLatticeValue &rhs) const {
    if (isUnknown())
      return *this;
    if (rhs.isUnknown())
      return rhs;
    if (*this == rhs)
      return *this;
    return TypeTagLatticeValue();
  }

private:
  ::std::optional<TypeTag> typeTag = ::std::nullopt;
};

class SparseTypeTagAnalysis
    : public SparseForwardDataFlowAnalysis<Lattice<TypeTagLatticeValue>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // TODO: implement this
  void
  visitOperation(Operation *op,
                 ArrayRef<const Lattice<TypeTagLatticeValue> *> operands,
                 ArrayRef<Lattice<TypeTagLatticeValue> *> results) override;

  // TODO: implement this
  void setToEntryState(Lattice<TypeTagLatticeValue> *lattice) override;
};

} // namespace mlir::lean

#endif