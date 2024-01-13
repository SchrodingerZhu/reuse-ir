#ifndef LEAN_DIALECT_H
#define LEAN_DIALECT_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Lean/IR/LeanOpsAttributes.h"
#include "Lean/IR/LeanOpsTypes.h"
#include "Refcnt/IR/RefcntOpsTypes.h"

#define GET_OP_CLASSES
#include "Lean/IR/LeanOps.h.inc"

#endif // LEAN_DIALECT_H
