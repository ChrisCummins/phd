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

namespace rewriter {

static clang::Rewriter rewriter;
static int numFunctions = 0;
static llvm::cl::OptionCategory _tool_category("phd");

class RewriterVisitor : public clang::RecursiveASTVisitor<RewriterVisitor> {
 private:
  std::unique_ptr<clang::ASTContext> astContext;  // additional AST info

  virtual ~RewriterVisitor() {}

 public:
  explicit RewriterVisitor(clang::CompilerInstance *CI)
      : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(),
                          astContext->getLangOpts());
  }

  virtual bool VisitFunctionDecl(clang::FunctionDecl *func) {
    numFunctions++;
    std::string funcName = func->getNameInfo().getName().getAsString();
    if (funcName == "do_math") {
      rewriter.ReplaceText(
          func->getLocation(),
          static_cast<unsigned int>(funcName.length()),
          "add5");
      llvm::errs() << "** Rewrote function def: " << funcName << "\n";
    }
    return true;
  }

  virtual bool VisitStmt(clang::Stmt *st) {
    if (clang::ReturnStmt *ret = clang::dyn_cast<clang::ReturnStmt>(st)) {
      rewriter.ReplaceText(ret->getRetValue()->getLocStart(), 6, "val");
      llvm::errs() << "** Rewrote ReturnStmt\n";
    }
    if (clang::CallExpr *call = clang::dyn_cast<clang::CallExpr>(st)) {
      rewriter.ReplaceText(call->getLocStart(), 7, "add5");
      llvm::errs() << "** Rewrote function call\n";
    }
    return true;
  }
  /*
    virtual bool VisitReturnStmt(ReturnStmt *ret) {
    rewriter.ReplaceText(ret->getRetValue()->getLocStart(), 6, "val");
    llvm::errs() << "** Rewrote ReturnStmt\n";
    return true;
    }

    virtual bool VisitCallExpr(CallExpr *call) {
    rewriter.ReplaceText(call->getLocStart(), 7, "add5");
    llvm::errs() << "** Rewrote function call\n";
    return true;
    }
  */
};


class RewriterASTConsumer : public clang::ASTConsumer {
 private:
  RewriterVisitor *visitor;

 public:
  // override the constructor in order to pass CI
  explicit RewriterASTConsumer(clang::CompilerInstance *CI)
      : visitor(new RewriterVisitor(CI))
  { }

  // override this to call our RewriterVisitor on the entire source file
  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
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


class RewriterFrontendAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &CI,
      StringRef file) {
    return llvm::make_unique<RewriterASTConsumer>(&CI);
  }
};


}  // namespace rewriter


int main(int argc, const char **argv) {
  clang::tooling::CommonOptionsParser op(argc, argv, rewriter::_tool_category);
  clang::tooling::ClangTool tool(op.getCompilations(), op.getSourcePathList());

  auto result = tool.run(
      clang::tooling::newFrontendActionFactory<
        rewriter::RewriterFrontendAction>().get());

  llvm::errs() << "\nFound " << rewriter::numFunctions << " functions.\n\n";
  const auto& id = rewriter::rewriter.getSourceMgr().getMainFileID();
  rewriter::rewriter.getEditBuffer(id).write(llvm::errs());
  return result;
}
