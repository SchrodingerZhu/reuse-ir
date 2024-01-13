  module {
    func.func @rev_map(%f: !refcnt.rc<!lean.obj>, %xs : !refcnt.rc<!lean.obj>, %acc : !refcnt.rc<!lean.obj>) -> !refcnt.rc<!lean.obj> 
      attributes { tag = "type-tag" }
    {
      cf.br ^start(%xs, %acc: !refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>)
    ^start(%list: !refcnt.rc<!lean.obj>, %updated : !refcnt.rc<!lean.obj>):
      %tag = lean.tag %list
      cf.switch %tag : i16, [
        default: ^bb_default,
        0: ^bb_nil,
        1: ^bb_cons
      ] { lean.type = "List" }
    ^bb_default:
      llvm.unreachable
    ^bb_nil:
      refcnt.dec %f : !refcnt.rc<!lean.obj>
      refcnt.dec %list : !refcnt.rc<!lean.obj>
      return %updated : !refcnt.rc<!lean.obj>
    ^bb_cons:
      %hd = lean.proj %list, 0 : index
      %tl = lean.proj %list, 1 : index
      refcnt.inc %hd : !refcnt.rc<!lean.obj>
      refcnt.inc %tl : !refcnt.rc<!lean.obj>
      refcnt.dec %list : !refcnt.rc<!lean.obj>
      refcnt.inc %f : !refcnt.rc<!lean.obj>
      %y = lean.app %f (%hd)
      %cons = refcnt.new { lean.type = "List" , lean.tag = 1 } : !refcnt.rc<!lean.obj>
      lean.set %cons, 0 : index, %y
      lean.set %cons, 1 : index, %updated
      cf.br ^start(%tl, %cons: !refcnt.rc<!lean.obj>, !refcnt.rc<!lean.obj>)
    }
}