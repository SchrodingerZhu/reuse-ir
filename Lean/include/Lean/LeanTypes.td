#ifndef LEAN_TYPES 
#define LEAN_TYPES

include "Lean/LeanDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Lean_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Lean_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // REFCNT_TYPES
