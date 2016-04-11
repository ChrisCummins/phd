// Usage:
//
//     LD_LIBRARY_PATH=~/phd/tools/llvm/build/lib ./rewriter <file> --
//
#include <memory>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#pragma GCC diagnostic pop

using namespace clang;  // NOLINT
using namespace clang::driver;  // NOLINT
using namespace clang::tooling;  // NOLINT
using namespace llvm;  // NOLINT

using namespace clang::tooling;  // NOLINT
using namespace llvm;  // NOLINT

Rewriter rewriter;
int numFunctions = 0;

static llvm::cl::OptionCategory _tool_category("phd");

class RewriterVisitor : public RecursiveASTVisitor<RewriterVisitor> {
 private:
  std::unique_ptr<ASTContext> astContext;  // additional AST info

  virtual ~RewriterVisitor() {}

 public:
  explicit RewriterVisitor(CompilerInstance *CI)
      : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(),
                          astContext->getLangOpts());
  }

  virtual bool VisitFunctionDecl(FunctionDecl *func) {
    numFunctions++;
    std::string funcName = func->getNameInfo().getName().getAsString();
    if (funcName == "do_math") {
      rewriter.ReplaceText(func->getLocation(),
                           static_cast<unsigned int>(funcName.length()),
                           "add5");
      errs() << "** Rewrote function def: " << funcName << "\n";
    }
    return true;
  }

  virtual bool VisitStmt(Stmt *st) {
    if (ReturnStmt *ret = dyn_cast<ReturnStmt>(st)) {
      rewriter.ReplaceText(ret->getRetValue()->getLocStart(), 6, "val");
      errs() << "** Rewrote ReturnStmt\n";
    }
    if (CallExpr *call = dyn_cast<CallExpr>(st)) {
      rewriter.ReplaceText(call->getLocStart(), 7, "add5");
      errs() << "** Rewrote function call\n";
    }
    return true;
  }
  /*
    virtual bool VisitReturnStmt(ReturnStmt *ret) {
    rewriter.ReplaceText(ret->getRetValue()->getLocStart(), 6, "val");
    errs() << "** Rewrote ReturnStmt\n";
    return true;
    }

    virtual bool VisitCallExpr(CallExpr *call) {
    rewriter.ReplaceText(call->getLocStart(), 7, "add5");
    errs() << "** Rewrote function call\n";
    return true;
    }
  */
};


class ExampleASTConsumer : public ASTConsumer {
 private:
  RewriterVisitor *visitor;

 public:
  // override the constructor in order to pass CI
  explicit ExampleASTConsumer(CompilerInstance *CI)
      : visitor(new RewriterVisitor(CI))
  { }

  // override this to call our RewriterVisitor on the entire source file
  virtual void HandleTranslationUnit(ASTContext &Context) {
    /* we can use ASTContext to get the TranslationUnitDecl, which is
       a single Decl that collectively represents the entire source file */
    visitor->TraverseDecl(Context.getTranslationUnitDecl());
  }

  /*
  // override this to call our RewriterVisitor on each top-level Decl
  virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
  // a DeclGroupRef may have multiple Decls, so we iterate through each one
  for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; i++) {
  Decl *D = *i;
  visitor->TraverseDecl(D); // recursively visit each AST node in Decl "D"
  }
  return true;
  }
  */
};


class RewriterFrontendAction : public ASTFrontendAction {
 public:
  virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                         StringRef file) {
    return make_unique<ExampleASTConsumer>(&CI);
  }
};


int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, _tool_category);
  ClangTool tool(op.getCompilations(), op.getSourcePathList());
  int result = tool.run(
      newFrontendActionFactory<RewriterFrontendAction>().get());

  errs() << "\nFound " << numFunctions << " functions.\n\n";
  rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(errs());
  return result;
}
