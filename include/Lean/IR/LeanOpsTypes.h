#ifndef LEAN_TYPES_H
#define LEAN_TYPES_H
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#define GET_TYPEDEF_CLASSES
#include "Lean/IR/LeanOpsTypes.h.inc"
#endif // LEAN_TYPES_H
