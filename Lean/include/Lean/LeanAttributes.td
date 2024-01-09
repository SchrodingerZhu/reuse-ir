#ifndef LEAN_ATTRIBUTES
#define LEAN_ATTRIBUTES

include "Lean/LeanDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

class Lean_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Lean_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // LEAN_ATTRIBUTES