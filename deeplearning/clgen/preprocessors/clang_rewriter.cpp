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
//
// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
//
// clgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clgen.  If not, see <https://www.gnu.org/licenses/>.
#define E_NO_INPUT 204

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>

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

// Uncomment the following line for verbose output:
// #define VERBOSE
// Uncommend the following line for debug output:
// #define DEBUG

// debugging print out
#ifdef DEBUG
# define DEBUG_OUT(x) llvm::errs() << x;
#else
# define DEBUG_OUT(x)
#endif


#ifndef REWRITE_STYLE
# warning "use -DREWRITE_STYLE to define a rewrite style"
# define REWRITE_STYLE 0
#endif

#if REWRITE_STYLE == 0
//
// function names
const std::string fn_prefix = "";
const char fn_base_char = 'A';
// variable names
const char var_base_char = 'a';
const std::string var_prefix = "";
const std::string gb_prefix = "G";
//
#elif REWRITE_STYLE == 1
//
// function names
const std::string fn_prefix = "fn_";
const char fn_base_char = 'A';
// variable names
const char var_base_char = 'a';
const std::string var_prefix = "";
const std::string gb_prefix = "gb_";
//
#else
#error "unknown rewrite style"
#endif


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


typedef std::map<std::string, std::string> rewrite_table_t;


// generate a new identifier name
//
// Takes an existing rewrite table and inserts a.
//
std::string get_next_name(rewrite_table_t& rewrites,
                          const std::string& name, const char& base_char,
                          const std::string& prefix = "") {
  auto i = rewrites.size();

  std::string s = prefix;

  // build the new name character by character
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
    return get_next_name(rewrites, name, base_char, prefix);
  }

  // insert the re-write name
  rewrites[name] = s;

  return s;
}


class RewriterVisitor : public clang::RecursiveASTVisitor<RewriterVisitor> {
 private:
  std::unique_ptr<clang::ASTContext> _context;  // additional AST info

  // identifier rewrite tables. There's one table to rewrite function names,
  // one table to rewrite global variables, and one table for each user
  // function:
  rewrite_table_t _fns;
  rewrite_table_t _global_vars;
  std::map<std::string, rewrite_table_t> _local_vars;

  // accepts a function name, and returns the rewritten name.
  //
  std::string get_fn_rewrite(const std::string& name) {
    if (_fns.find(name) == _fns.end()) {
      // New function:
      auto replacement = get_next_name(_fns, name, fn_base_char, fn_prefix);
      return replacement;
    } else {
      // Previously declared function:
      return (*_fns.find(name)).second;
    }
  }

  // accepts a variable name, and returns the rewritten name
  //
  std::string get_var_rewrite(rewrite_table_t& rewrites,
                              const std::string& name,
                              const std::string& prefix="") {
    if (rewrites.find(name) == rewrites.end()) {
      // New variable:
      auto replacement = get_next_name(rewrites, name, var_base_char, prefix);
      return replacement;
    } else {
      // Previously declared variable:
      return (*rewrites.find(name)).second;
    }
  }

  // accepts a function declaration, and returns the rewrite table for it
  //
  rewrite_table_t& get_fn_var_rewrite_table(const clang::FunctionDecl* fn) {
    const auto fn_name = fn->getNameInfo().getName().getAsString();
    const auto fn_it = _local_vars.find(fn_name);

    // if there's no existing table, create one
    if (fn_it == _local_vars.end())
      _local_vars[fn_name] = rewrite_table_t();

    return _local_vars[fn_name];
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

  void rewrite_fn_name(clang::FunctionDecl *const func,
                       const std::string& replacement) {
   rewriter.ReplaceText(func->getLocation(), replacement);
   ++_fn_call_rewrites_counter;
  }

  void rewrite_fn_name(clang::CallExpr *const call,
                       const std::string& replacement) {
   rewriter.ReplaceText(call->getLocStart(), replacement);
   ++_fn_call_rewrites_counter;
  }

  void rewrite_var_name(clang::DeclRefExpr *const ref,
                        const std::string& replacement) {
    rewriter.ReplaceText(ref->getLocStart(), replacement);
    ++_var_use_rewrites_counter;
  }

  void rewrite_var_name(clang::VarDecl *const decl,
                        const std::string& replacement) {
    rewriter.ReplaceText(decl->getLocation(), replacement);
    ++_var_decl_rewrites_counter;
  }

  // rewrite function declarations
  //
  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    // only re-write functions declared in the main file
    if (isMainFile(func->getLocation())) {
      const auto name = func->getNameInfo().getName().getAsString();
      const auto replacement = get_fn_rewrite(name);
      rewrite_fn_name(func, replacement);
      DEBUG_OUT("FunctionDecl " << name << " -> " << replacement << '\n');
    }

    return true;
  }

  // rewrite function calls
  //
  bool VisitCallExpr(clang::CallExpr *call) {
    if (isMainFile(call->getLocStart())) {
      // rewrite function calls
      const auto callee = call->getDirectCallee();
      if (callee) {
        const auto name = callee->getNameInfo().getName().getAsString();

        // rewrite fn name
        const auto it = _fns.find(name);
        if (it != _fns.end()) {
          const auto replacement = (*it).second;
          rewrite_fn_name(call, replacement);
          DEBUG_OUT("CallExpr " << name << " -> " << replacement << '\n');
        }
      }  // else not a direct callee (do we need to handle that?)
    }  // else not in main file

    return true;
  }

  // rewrite variable declarations
  //
  bool VisitVarDecl(clang::VarDecl *decl) {
    // only re-write variables declared in the main file
    if (!isMainFile(decl->getLocation()))
      return true;

    if (auto d = clang::dyn_cast<clang::NamedDecl>(decl)) {
      const auto name = d->getNameAsString();

      // variables can be declared without a name (e.g. in function
      // declarations). Do not rewrite these
      if (name.empty())
        return true;

      // get the parent function
      const auto* parent = d->getParentFunctionOrMethod();
      if (parent == nullptr) {
        // if there's no parent, then it's a global variable
        const auto replacement = get_var_rewrite(
            _global_vars,  // rewrite table
            name,  // original name
            gb_prefix);  // prefix for new name

        // rewrite variable name
        rewrite_var_name(decl, replacement);
        DEBUG_OUT("VarDecl " << name << " -> " << replacement << '\n');
      } else if (auto fn = clang::dyn_cast<clang::FunctionDecl>(parent)) {
        // if it's in function scope, get the rewrite table
        auto& rewrite_table = get_fn_var_rewrite_table(fn);
        const auto replacement = get_var_rewrite(
            rewrite_table, // rewrite table
            name,  // original name
            var_prefix);  // prefix for new name

        // rewrite variable name
        rewrite_var_name(decl, replacement);
        DEBUG_OUT("VarDecl " << name << " -> " << replacement << '\n');
      } else {
        // this shouldn't happen
        llvm::errs() << "warning: cannot determine scope of variable '"
                     << name << "'\n";
      }
    }

    return true;
  }

  // rewrite variable refs
  //
  bool VisitDeclRefExpr(clang::DeclRefExpr* ref) {
    if (isMainFile(ref->getLocStart())) {
      const auto name = ref->getNameInfo().getName().getAsString();
      const auto d = ref->getDecl();

      const auto* parent = d->getParentFunctionOrMethod();
      if (parent == nullptr) {
        // get rewrite name
        const auto it = _global_vars.find(name);

        // rewrite
        if (it != _global_vars.end()) {
          const auto replacement = (*it).second;
          rewrite_var_name(ref, replacement);
          DEBUG_OUT("DeclRefExpr " << name << " -> " << replacement << '\n');
        }
      } else if (auto fn = clang::dyn_cast<clang::FunctionDecl>(parent)) {
        // get rewrite name
        const auto& lookup_table = get_fn_var_rewrite_table(fn);
        const auto it = lookup_table.find(name);

        // rewrite
        if (it != lookup_table.end()) {
          const auto replacement = (*it).second;
          rewrite_var_name(ref, replacement);
          DEBUG_OUT("DeclRefExpr " << name << " -> " << replacement << '\n');
        }
      } else {
        llvm::errs() << "warning: cannot determine scope of variable '" << name << "'\n";
      }
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
