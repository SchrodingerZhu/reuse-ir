#ifndef REFCNT_OPS
#define REFCNT_OPS

include "Refcnt/RefcntDialect.td"
include "Refcnt/RefcntTypes.td"

class Refcnt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Refcnt_Dialect, mnemonic, traits>;

#endif // REFCNT_OPS
