#ifndef REFCNT_TYPES 
#define REFCNT_TYPES

include "Refcnt/RefcntDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Refcnt_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Refcnt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Refcnt_RcType : Refcnt_Type<"Rc", "rc",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "RC pointer type";
  let description = [{
    `Refcnt.rc` is the type of a reference-counted pointer.
  }];

  let parameters = (ins "::mlir::Type":$pointee);

  let assemblyFormat = "`<` $pointee `>`";
}

def Refcnt_TokenType : Refcnt_Type<"Token", "token",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "Reuse token type";
  let description = [{
    `Refcnt.token` is the type of a memory token.
  }];

  let parameters = (ins "::mlir::Type":$layout);

  let assemblyFormat = "`<` $layout `>`";
}

#endif // REFCNT_TYPES
