/*
 * An implementation of the feature extractor used in:
 *
 *     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
 *     mapping of data parallel programs to OpenCL for heterogeneous
 *     systems. In CGO. IEEE.
 *
 * Written by Zheng Wang <zh.wang@ed.ac.uk>.
 */
#ifndef __FEATURE_EXTRACTOR_H__
#define __FEATURE_EXTRACTOR_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <sstream>

#include "llvm/Support/FileSystem.h"

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTContext.h"


#endif
