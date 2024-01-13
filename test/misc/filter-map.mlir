  module {
    func.func @filter_map(%f: !refcnt.rc<!lean.obj>, %xs : !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj> 
      attributes { tag = "type-tag" }
    {
      %tag = lean.tag %xs
      cf.switch %tag : i16, [
        default: ^bb_default,
        0: ^bb_nil,
        1: ^bb_cons
      ] { lean.type = "List" }
    ^bb_nil:
      refcnt.dec %f : !refcnt.rc<!lean.obj>
      return %xs : !refcnt.rc<!lean.obj>
    ^bb_cons:
      // project all the fields
      %hd = lean.proj %xs, 0 : index
      %tl = lean.proj %xs, 1 : index
      refcnt.inc %hd : !refcnt.rc<!lean.obj>
      refcnt.inc %tl : !refcnt.rc<!lean.obj>
      refcnt.dec %xs : !refcnt.rc<!lean.obj>
      // prepare for application
      refcnt.inc %f : !refcnt.rc<!lean.obj>
      refcnt.inc %hd : !refcnt.rc<!lean.obj>
      %chk = lean.app %f (%hd)
      // todo: change this to unbox
      %t = lean.tag %chk
      cf.switch %t : i16, [
        default: ^bb_default,
        0: ^bb_false,
        1: ^bb_true
      ] { lean.type = "Bool" }
    ^bb_false:
      refcnt.dec %chk : !refcnt.rc<!lean.obj>
      refcnt.dec %hd : !refcnt.rc<!lean.obj>
      %r = func.call @filter_map(%f, %tl) : (!refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj>
      return %r : !refcnt.rc<!lean.obj>
    ^bb_true:
      refcnt.dec %chk : !refcnt.rc<!lean.obj>
      %ys = func.call @filter_map(%f, %tl) : (!refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj>
      %cons = refcnt.new { lean.type = "List" , lean.tag = 1 } : !refcnt.rc<!lean.obj>
      lean.set %cons, 0 : index, %hd
      lean.set %cons, 1 : index, %ys
      return %cons : !refcnt.rc<!lean.obj>
    ^bb_default:
      llvm.unreachable
    }
}