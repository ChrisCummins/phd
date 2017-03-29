// rewriter: Rewrites source code for machine learning
//
// Rewrite all the names of names of variables and functions declared
// in an input source to be as short as possible.
//
// Usage:
//
//     ./rewriter foo.cl \
//         -extra-arg=-Dcl_clang_storage_class_specifiers \
//         -extra-arg=-I/path/to/libclc/generic/include \
//         -extra-arg=-include \
//         -extra-arg=/path/to/libclc/generic/include/clc/clc.h \
//         -extra-arg=-target -extra-arg=nvptx64-nvidia-nvcl \
//         -extra-arg=-xcl --
//
// Prints the number of variable and function names that are rewritten. If
// nothing is rewritten, exit with status code
#define E_NO_INPUT 204
//
// Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
//
// This file is part of CLgen.
//
// CLgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CLgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
//
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>


// Uncomment the following line for verbose output:
// #define VERBOSE


namespace rewriter {

// global state
static clang::Rewriter rewriter;

static llvm::cl::OptionCategory _tool_category("clgen");

// function rewrite counters
static unsigned int _fn_decl_rewrites_counter = 0;
static unsigned int _fn_call_rewrites_counter = 0;

// variable rewrite counters
static unsigned int _var_decl_rewrites_counter = 0;
static unsigned int _var_use_rewrites_counter = 0;


// determine if rewriter has done any rewriting
//
static bool isRewritten() {
  return (rewriter::_fn_decl_rewrites_counter ||
          rewriter::_fn_call_rewrites_counter ||
          rewriter::_var_decl_rewrites_counter ||
          rewriter::_var_use_rewrites_counter);
}


// character types
enum ctype { AZ, az };

// reserved OpenCL keywords to check for when picking a name. Of course, there
// are many reserved words, but we select only the short ones, which are
// plausible to occur in the real world.
//
std::set<std::string> reserved_names {
  "do",
  "if",
  "abs",
  "for",
  "int"
};


// generate a new identifier name
//
// Takes an existing rewrite table and inserts a.
//
std::string get_next_name(std::map<std::string, std::string>& rewrites,
                          const std::string& name, const char& base_char,
                          const std::string& prefix = "") {
  auto i = rewrites.size();

  std::string s = prefix;

  // build the new name character by character.
  while (i > 25) {
    auto k = i / 26;
    i %= 26;
    s.push_back(base_char - 1 + k);
  }
  s.push_back(base_char + i);

  // Check that it isn't a reserved word, or else generate a new one
  if (reserved_names.find(s) != reserved_names.end()) {
    // insert a "dummy" value using an illegal identifier name.
    std::stringstream invalid_identifier;
    invalid_identifier << "\t!@invalid@!\t" << rewrites.size();

    rewrites[invalid_identifier.str()] = s;
    return get_next_name(rewrites, name, base_char);
  }

  // insert the re-write name
  rewrites[name] = s;

  return s;
}


class RewriterVisitor : public clang::RecursiveASTVisitor<RewriterVisitor> {
 private:
  std::unique_ptr<clang::ASTContext> _context;  // additional AST info

  // function name rewriting
  std::map<std::string, std::string> _fns;

  // variable name rewriting
  std::map<std::string, std::string> _vars;

  // accepts a function name, and returns the rewritten name.
  //
  std::string get_fn_rewrite(const std::string& name) {
    if (_fns.find(name) == _fns.end()) {
      // New function:
      auto replacement = get_next_name(_fns, name, 'A', "fn_");
      return replacement;
    } else {
      // Previously declared function:
      return (*_fns.find(name)).second;
    }
  }

  // accepts a variable name, and returns the rewritten name
  //
  std::string get_var_rewrite(std::map<std::string, std::string>& rewrites,
                              const std::string& name,
                              const std::string& prefix="") {
    if (rewrites.find(name) == rewrites.end()) {
      // New variable:
      auto replacement = get_next_name(rewrites, name, 'A');
      return replacement;
    } else {
      // Previously declared variable:
      return (*rewrites.find(name)).second;
    }
  }

  // return true if a location is in the main source file
  //
  bool isMainFile(const clang::SourceLocation& location) {
    auto& srcMgr = _context->getSourceManager();
    const auto file_id = srcMgr.getFileID(location).getHashValue();

    return file_id == 1;
  }

 public:
  explicit RewriterVisitor(clang::CompilerInstance *ci)
      : _context(&(ci->getASTContext())) {
    rewriter.setSourceMgr(_context->getSourceManager(),
                          _context->getLangOpts());
  }

  virtual ~RewriterVisitor() {}

  // rewrite function declarations
  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    if (!isMainFile(func->getLocation()))
      return true;

    const auto name = func->getNameInfo().getName().getAsString();
    const auto replacement = get_fn_rewrite(name);

    rewriter.ReplaceText(func->getLocation(), replacement);
    ++_fn_decl_rewrites_counter;

    return true;
  }

  // rewrite variable declarations
  bool VisitVarDecl(clang::VarDecl *decl) {
    if (!isMainFile(decl->getLocation()))
      return true;

    if (auto d = clang::dyn_cast<clang::NamedDecl>(decl)) {
      const auto name = d->getNameAsString();

      // variables can be declared without a name (e.g. in function
      // declarations). Do not rewrite these names.
      if (name.empty())
        return true;

      const auto replacement = get_var_rewrite(_vars, name);
      rewriter.ReplaceText(decl->getLocation(), replacement);
      ++_var_decl_rewrites_counter;
    }

    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr* ref) {
    if (isMainFile(ref->getLocStart())) {
      const auto name = ref->getNameInfo().getName().getAsString();

      const auto it = _global_vars.find(name);
      if (it != _global_vars.end()) {
        const auto replacement = (*it).second;
        rewriter.ReplaceText(ref->getLocStart(), replacement);
        ++_var_use_rewrites_counter;
      }  // else variable name is externally defined
    }  // else not in main file

    return true;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (isMainFile(call->getLocStart())) {
      // rewrite function calls
      const auto callee = call->getDirectCallee();
      if (callee) {
        const auto name = callee->getNameInfo().getName().getAsString();
        const auto it = _fns.find(name);
        if (it != _fns.end()) {
          const auto replacement = (*it).second;

          rewriter.ReplaceText(call->getLocStart(), replacement);
          ++_fn_call_rewrites_counter;
        }  // else function name is externally defined
      }  // else not a direct callee (do we need to handle that?)
    }  // else not in main file

    return true;
  }
};


class RewriterASTConsumer : public clang::ASTConsumer {
 private:
  RewriterVisitor *visitor;

 public:
  // override the constructor in order to pass CI
  explicit RewriterASTConsumer(clang::CompilerInstance *ci)
      : visitor(new RewriterVisitor(ci))
  { }

  // override this to call our RewriterVisitor on the entire source file
  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    // use ASTContext to get the TranslationUnitDecl, which is
    // a single Decl that collectively represents the entire source file
    visitor->TraverseDecl(Context.getTranslationUnitDecl());
  }
};


class RewriterFrontendAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& ci, StringRef file) {
    return llvm::make_unique<RewriterASTConsumer>(&ci);
  }
};


}  // namespace rewriter


// let's get shit done!
//
int main(int argc, const char** argv) {
  clang::tooling::CommonOptionsParser op(argc, argv, rewriter::_tool_category);
  clang::tooling::ClangTool tool(op.getCompilations(), op.getSourcePathList());

  const auto result = tool.run(
      clang::tooling::newFrontendActionFactory<
        rewriter::RewriterFrontendAction>().get());

  const auto& id = rewriter::rewriter.getSourceMgr().getMainFileID();

#ifdef VERBOSE
  if (!rewriter::isRewritten()) {
    llvm::errs() << "fatal: nothing to rewrite!";
    return E_NO_INPUT;
  }

  llvm::errs() << "\nRewrote " << rewriter::_fn_decl_rewrites_counter
               << " function declarations\n"
               << "Rewrote " << rewriter::_fn_call_rewrites_counter
               << " function calls\n\n"
               << "Rewrote " << rewriter::_var_decl_rewrites_counter
               << " variable declarations\n"
               << "Rewrote " << rewriter::_var_use_rewrites_counter
               << " variable uses\n";
#else  // not VERBOSE
  if (!rewriter::isRewritten())
    return E_NO_INPUT;
#endif  // VERBOSE

  rewriter::rewriter.getEditBuffer(id).write(llvm::outs());
  return result;
}
