#ifndef REFCNT_DIALECT
#define REFCNT_DIALECT

include "mlir/IR/OpBase.td"

def Refcnt_Dialect : Dialect {
  let name = "refcnt";
  let cppNamespace = "::mlir::refcnt";
  let description = [{
    The `refcnt` dialect is designed for languages using internalized RC to manage
    their memory. This dialect provides a core set of RC related operations. Utilizing
    `reuse analysis`, the dialect can also provide a set of optimizations to reduce
    extraneous memory allocations and deallocations.
  }];
  let dependentDialects = [];
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    private:
      void addTypesImpl();
      void addOpsImpl();
  }];
  //let hasConstantMaterializer = 1;
}

#endif // REFCNT_DIALECT
