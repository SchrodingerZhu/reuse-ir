#ifndef REFCNT_TYPES 
#define REFCNT_TYPES

include "Refcnt/RefcntDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Refcnt_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Refcnt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // REFCNT_TYPES