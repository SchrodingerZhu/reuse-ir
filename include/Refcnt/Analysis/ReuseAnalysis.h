#ifndef REFCNT_ANALYSIS_REUSE_ANALYSIS_H
#define REFCNT_ANALYSIS_REUSE_ANALYSIS_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>

#include "Refcnt/Utilities/ADT.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::refcnt {
class Reusable {
public:
  /// Return reusibility estimation of current value.
  /// Negative value means that current value is not reusable.
  virtual size_t reusability() const = 0;
  virtual void dump(llvm::raw_ostream &os, AsmState &) const;
  virtual ~Reusable() = default;
  const Value &getValue() const { return value; }
  Value &getValue() { return value; }
  TypeID getKind() const { return kind; }
  Reusable(TypeID kind, Value value) : kind(kind), value(std::move(value)) {}

private:
  TypeID kind;
  Value value;
};

struct ReusabilityComparator {
  bool operator()(const std::unique_ptr<Reusable> &lhs,
                  const std::unique_ptr<Reusable> &rhs) const {
    return lhs->reusability() > rhs->reusability();
  }
};

class ReusabibilityLookupTable
    : public AnalysisState,
      public refcnt::OrderedSet<std::unique_ptr<Reusable>,
                                ReusabilityComparator> {
public:
  Reusable *getMostPreferredReusable() const;
  void print(llvm::raw_ostream &os) const override;
};

/*
class ReuseToken {
public:
  enum class Kind {
    Conditional,
    Direct,
  };
  Kind getKind() const { return kind; }
  virtual Value getReusedValue() const = 0;
  virtual double getReuseProbability(DataFlowSolver &solver) const = 0;
  virtual ~ReuseToken() = default;
  ReuseToken(Kind kind) : kind(kind) {}

private:
  Kind kind;
};

struct ReuseDecision : public AnalysisState {
  ReuseToken *token;
  Reusable *reusable;
  double score(const ReusabibilityLookupTable &table,
               DataFlowSolver &solver) const {
    auto prob = token->getReuseProbability(solver);
    auto reusability = reusable->reusability();
    return prob * static_cast<double>(reusability);
  }
};

class ReuseTokenCollection : public AnalysisState,
                             public llvm::SmallDenseSet<ReuseToken *> {
public:
  void print(llvm::raw_ostream &os) const override;
};

class ConditonalReuseToken : public ReuseToken {
public:
  ConditonalReuseToken(Block *joinBlock, ReuseToken *source)
      : ReuseToken(Kind::Conditional), joinBlock(joinBlock), source(source) {}

  static bool classof(const ReuseToken *token) {
    return token->getKind() == Kind::Conditional;
  }

  Block *getJoinBlock() const { return joinBlock; }
  ReuseToken *getSource() const { return source; }
  Value getReusedValue() const override { return source->getReusedValue(); }
  double getReuseProbability(DataFlowSolver &solver) const override;

private:
  Block *joinBlock;
  ReuseToken *source;
};

class DirectReuseToken : public ReuseToken {
public:
  DirectReuseToken(Value value) : ReuseToken(Kind::Direct), value(value) {}

  static bool classof(const ReuseToken *token) {
    return token->getKind() == Kind::Direct;
  }

  Value getReusedValue() const override { return value; }
  double getReuseProbability(DataFlowSolver &) const override { return 1.0; }

private:
  Value value;
};
*/

} // namespace mlir::refcnt
#endif // REFCNT_ANALYSIS_REUSE_ANALYSIS_H