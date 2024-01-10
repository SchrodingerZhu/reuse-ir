#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>

#include "Refcnt/RefcntOpsDialect.h"
#include "Refcnt/RefcntOpsTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Refcnt/RefcntOpsTypes.cpp.inc"

namespace mlir::refcnt {
#define LAYOUT_AS_INDEX_TYPE(TYPE)                                             \
  ::llvm::TypeSize TYPE::getTypeSizeInBits(                                    \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeSizeInBits(::mlir::IndexType::get(getContext())); \
  }                                                                            \
  uint64_t TYPE::getABIAlignment(                                              \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeABIAlignment(                                     \
        ::mlir::IndexType::get(getContext()));                                 \
  }                                                                            \
  uint64_t TYPE::getPreferredAlignment(                                        \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypePreferredAlignment(                               \
        ::mlir::IndexType::get(getContext()));                                 \
  }

LAYOUT_AS_INDEX_TYPE(RcType)
LAYOUT_AS_INDEX_TYPE(TokenType)

void RefcntDialect::addTypesImpl() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Refcnt/RefcntOpsTypes.cpp.inc"
      >();
}
} // namespace mlir::refcnt