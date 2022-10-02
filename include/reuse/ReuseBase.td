#ifndef REUSE_BASE
#define REUSE_BASE

include "mlir/IR/OpBase.td"

def Reuse_Dialect : Dialect {
  let name = "reuse";

  let summary = "A Intermediate Representation for Reuse Analysis";

  let description = [{
    This IR conains basic RC runtime operations, providing analysis passes
    for memory reuse
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "reuse";

  let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

#endif // REUSE_BASE
