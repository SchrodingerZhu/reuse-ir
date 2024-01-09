#ifndef LEAN_DIALECT
#define LEAN_DIALECT

include "mlir/IR/OpBase.td"

def Lean_Dialect : Dialect {
  let name = "lean";
  let cppNamespace = "::mlir::lean";
  let description = [{
    The `lean` dialect is a low-level IR for the Lean programming language. It defines basic
    operations on Lean objects, such as allocation, reference counting, and boxing.
  }];
  let dependentDialects =
      ["::mlir::refcnt::RefcntDialect", "::mlir::BuiltinDialect"];
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // LEAN_DIALECT
