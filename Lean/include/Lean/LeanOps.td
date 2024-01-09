#ifndef LEAN_OPS
#define LEAN_OPS
include "Lean/LeanDialect.td"
include "Lean/LeanTypes.td"
include "Refcnt/RefcntOps.td"

class Lean_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lean_Dialect, mnemonic, traits>;

#endif // LEAN_OPS
