#ifndef REFCNT_DIALECT_H
#define REFCNT_DIALECT_H

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

#include "Refcnt/IR/RefcntOpsTypes.h"

#define GET_OP_CLASSES
#include "Refcnt/IR/RefcntOps.h.inc"

#endif // REFCNT_DIALECT_H
