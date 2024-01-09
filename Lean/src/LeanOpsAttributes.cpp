
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/LeanOpsAttributes.h"
#include "Lean/LeanOpsDialect.h"

#define GET_ATTRDEF_CLASSES
#include "Lean/LeanOpsAttributes.cpp.inc"