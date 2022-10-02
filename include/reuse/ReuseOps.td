#ifndef REUSE_OPS
#define REUSE_OPS

include "ReuseBase.td"
include "ReuseTypes.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"

class Reuse_Op<string mnemonic, list<Trait> traits = []> :
    Op<Reuse_Dialect, mnemonic, traits>;


def ReuseCreateOp : Reuse_Op<"create"> {
    let arguments = (ins AnyMemRef:$memref, BoolAttr:$atomic);
    let results = (outs Reuse_RcPtrType:$result);
    let builders = [
        OpBuilder<
            (ins "::mlir::Value":$memref, "::reuse::ObjectType":$object, "bool":$atomic),
            [{
                auto attr = $_builder.getBoolAttr(atomic);
                auto result = ::reuse::RcPtrType::get($_builder.getContext(), object, attr);

                $_state.addOperands(memref);
                $_state.addAttribute("atomic", attr);
                $_state.addTypes(result);
            }]
        >
    ];
    let assemblyFormat = [{
        `(` $memref `)` attr-dict `:` type($result) `[` type($memref) `]`
    }];
    // let hasCanonicalizer = 1;
    // let hasVerifier = 1;
}

// def ReuseDecrefOp : Reuse_Op<"decref"> {
    
// }

// def ReuseIncrefOp : Reuse_Op<"incref"> {
    
// }

// def ReuseIncrefOp : Reuse_Op<"release"> {
    
// }

#endif // REUSE_OPS