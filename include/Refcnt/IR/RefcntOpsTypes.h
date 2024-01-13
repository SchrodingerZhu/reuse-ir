#ifndef REFCNT_TYPES_H
#define REFCNT_TYPES_H
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#define GET_TYPEDEF_CLASSES
#include "Refcnt/IR/RefcntOpsTypes.h.inc"
#endif // REFCNT_TYPES_H
