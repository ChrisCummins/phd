// Work in progress on Java AST rewriter.
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
package deeplearning.clgen.preprocessors;

import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Set;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.ToolFactory;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.ArrayAccess;
import org.eclipse.jdt.core.dom.ArrayCreation;
import org.eclipse.jdt.core.dom.Assignment;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.IfStatement;
import org.eclipse.jdt.core.dom.InfixExpression;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.PostfixExpression;
import org.eclipse.jdt.core.dom.PrefixExpression;
import org.eclipse.jdt.core.dom.QualifiedName;
import org.eclipse.jdt.core.dom.ReturnStatement;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.SimpleType;
import org.eclipse.jdt.core.dom.SingleVariableDeclaration;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.formatter.CodeFormatter;
import org.eclipse.jdt.internal.compiler.impl.CompilerOptions;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.TextEdit;

/**
 * A rewriter for Java source code.
 */
public class JavaRewriter {

  private static final String[] RESERVED_WORDS_ = new String[]{
      "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
      "class", "const", "continue", "default", "do", "double", "else", "enum",
      "extends", "false", "final", "finally", "float", "for", "goto", "if",
      "implements", "import", "instanceof", "int", "interface", "long",
      "native", "new", "null", "package", "private", "protected", "public",
      "return", "short", "static", "strictfp", "super", "switch",
      "synchronized", "this", "throw", "throws", "transient", "true", "try",
      "void", "volatile", "while",
  };
  private static final Set<String> RESERVED_WORDS = new HashSet<>(
      Arrays.asList(RESERVED_WORDS_));
  private ASTRewrite traversalRewrite;
  private AST traversalAST;
  private HashMap<String, String> methodRewrites = new HashMap<>();
  private HashMap<String, String> typeRewrites = new HashMap<>();
  private HashMap<String, String> fieldRewrites = new HashMap<>();
  private HashMap<String, String> variableRewrites = new HashMap<>();

  /**
   * Strip the comments from a compilation unit.
   *
   * This iterates *backwards* over the comments, creating a DeleteEdit instance
   * for each comment. These edits must be applied in the order they are
   * returned, since we use simple indexing into the string source to determine
   * where to snip the code.
   *
   * @param edits The edit list to append comment-removing edits to.
   * @param compilationUnit The compilation unit to remove comments from.
   */
  private static void StripComments(ArrayList<TextEdit> edits,
      final CompilationUnit compilationUnit) {
    final List comments = compilationUnit.getCommentList();
    ListIterator li = comments.listIterator(comments.size());
    while (li.hasPrevious()) {
      Object object = li.previous();
      if (object instanceof Comment) {
        final int start = ((Comment) object).getStartPosition();
        final int length = ((Comment) object).getLength();
        edits.add(new DeleteEdit(start, length));
      }
    }
  }

  /**
   * Format a Java source code.
   *
   * @param source The source code to format.
   * @return The formatted source code.
   */
  private static String FormatSource(final String source) {
    Document document = new Document(source);
    ArrayList<TextEdit> edits = new ArrayList<>();
    Properties codeFormatterProperties = new Properties();
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_SOURCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_COMPLIANCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, CompilerOptions.VERSION_1_8);
    // Allow really, really long lines.
    codeFormatterProperties.setProperty(
        "org.eclipse.jdt.core.formatter.lineSplit", "9999");
    codeFormatterProperties.setProperty(
        "org.eclipse.jdt.core.formatter.comment.line_length", "9999");
    CodeFormatter codeFormatter = ToolFactory.createCodeFormatter(
        codeFormatterProperties);
    edits.add(codeFormatter.format(
        CodeFormatter.K_COMPILATION_UNIT, source, 0, source.length(),
        0, null));
    ApplyEdits(edits, document);
    return document.get();
  }

  /**
   * Get the compilation unit for a document.
   *
   * @param document The document to get the compilation unit for.
   * @return The compilation unit.
   */
  private static CompilationUnit GetCompilationUnit(final Document document) {
    ASTParser parser = ASTParser.newParser(AST.JLS10);
    parser.setSource(document.get().toCharArray());
    return (CompilationUnit) parser.createAST(null);
  }

  /**
   * Apply a list of edits to a document.
   *
   * @param edits The edits to apply.
   * @param document The document to apply the edits to.
   */
  private static void ApplyEdits(final List<TextEdit> edits,
      Document document) {
    for (TextEdit edit : edits) {
      System.err.println("Applying edit: " + edit.toString());
      try {
        edit.apply(document);
      } catch (BadLocationException e) {
        System.out.println("Failed to apply text edit!");
      }
    }
  }

  /**
   * Read a file and return its contents as a string.
   *
   * @param path The path of the file to read.
   * @param encoding The encoding of the file.
   * @return A string of the file contents.
   * @throws IOException In case of IO error.
   */
  private static String ReadFile(String path, Charset encoding)
      throws IOException {
    byte[] encoded = Files.readAllBytes(Paths.get(path));
    return new String(encoded, encoding);
  }

  public static void main(final String[] args) {
    JavaRewriter rewriter = new JavaRewriter();

    try {
      // final String input =ReadFile(args[0], Charset.defaultCharset());
      final String input = new String(ByteStreams.toByteArray(System.in));
      final String source = rewriter.RewriteSource(input);
      if (source == null) {
        System.out.println("fatal: RewriteSource() returned null.");
        System.exit(1);
      }
      System.out.println(source);
    } catch (IOException e) {
      System.err.println("fatal: I/O error");
      System.exit(1);
    }
  }

  /**
   * Generate a new rewrite name.
   *
   * @param rewrites The rewrite table. The new name is added to this table.
   * @param name The current name.
   * @param base_char The base character to use for generating new names, e.g.
   * 'A' to produce the sequence 'A', 'B', 'C'...
   * @param name_prefix An optional prefix to prepend to generated names.
   * @return The replacement name.
   */
  private String GetNextName(HashMap<String, String> rewrites,
      final String name, final char base_char,
      final String name_prefix) {
    int i = rewrites.size();
    StringBuilder s = new StringBuilder();

    s.append(name_prefix);
    // Build the new name character by character
    while (i > 25) {
      int k = i / 26;
      i %= 26;
      s.append((char) (base_char - 1 + k));
    }
    s.append((char) (base_char + i));

    final String newName = s.toString();

    // Check that it isn't a reserved word, or else generate a new one.
    if (RESERVED_WORDS.contains(newName)) {
      // Insert a "dummy" value using an illegal identifier name.
      s.append("\t!@invalid@!\t");
      s.append(rewrites.size());
      final String invalidIdentifier = s.toString();
      rewrites.put(invalidIdentifier, invalidIdentifier);
      return GetNextName(rewrites, name, base_char, name_prefix);
    }

    // insert the re-write name
    rewrites.put(name, newName);
    return newName;
  }

  /**
   * Rewrite a Java source file to make it more amenable to machine learning.
   *
   * @param source The source code to rewrite.
   * @return The rewritten source code.
   */
  public String RewriteSource(String source) {
    Document document = new Document(source);
    CompilationUnit compilationUnit = GetCompilationUnit(document);
    ArrayList<TextEdit> edits = new ArrayList<>();
    // First strip the comments.
    StripComments(edits, compilationUnit);
    ApplyEdits(edits, document);
    source = document.get();
    edits.clear();
    // Rewrite the source code.
    document = new Document(source);
    compilationUnit = GetCompilationUnit(document);
    // Foo(edits, compilationUnit, document);
    RewriteIdentifiers(edits, compilationUnit, document);
    ApplyEdits(edits, document);
    // Format the source code.
    return FormatSource(document.get());
  }

  private void RewriteIdentifiers(ArrayList<TextEdit> edits,
      final CompilationUnit compilationUnit,
      final Document document) {
    this.traversalRewrite = ASTRewrite.create(compilationUnit.getAST());
    this.traversalAST = compilationUnit.getAST();

    // Rewrite declarations.

    compilationUnit.accept(new ASTVisitor() {
      /* Private helper method for renaming declarations. */
      private boolean Rename(final String type_name,
          final HashMap<String, String> rewriteTable,
          final ASTNode node, final char base_char, final String name_prefix) {
        final String oldName = node.toString();
        final String newName = GetNextName(rewriteTable, oldName, base_char,
            name_prefix);
        traversalRewrite.replace(
            node, traversalAST.newSimpleName(newName), null);
        System.err.println(
            "=> " + type_name + ": " + oldName + " -> " + newName);
        return true;
      }

      public boolean visit(TypeDeclaration node) {
        return Rename("TypeDeclaration", typeRewrites, node.getName(), 'A', "");
      }

      public boolean visit(MethodDeclaration node) {
        if (!node.getName().toString().equals("main") &&
            !node.isConstructor()) {
          Rename("MethodDeclaration", methodRewrites, node.getName(), 'A',
              "fn_");
        }
        return true;
      }

      public boolean visit(SingleVariableDeclaration node) {
        return Rename("SingleVariableDeclaration", variableRewrites,
            node.getName(), 'a', "");
      }

      public boolean visit(VariableDeclarationFragment node) {
        return Rename("VariableDeclarationFragment", variableRewrites,
            node.getName(), 'a', "");
      }
    });

    // Rewrite usages.

    compilationUnit.accept(new ASTVisitor() {

      /* Private helper method to rename usages. */
      private void Rename(final String type_name,
          final HashMap<String, String> rewriteTable,
          final ASTNode node) {
        final String oldName = node.toString();
        if (rewriteTable.containsKey(oldName)) {
          final String newName = rewriteTable.get(oldName);
          traversalRewrite.replace(
              node, traversalAST.newSimpleName(newName), null);
          System.err.println(
              "=> " + type_name + ": " + oldName + " -> " + newName);
        } else {
          System.err.println("!! " + type_name + ": miss for " + oldName);
        }
      }

      public boolean visit(SimpleName node) {
        final ASTNode parent = node.getParent();

        if (parent instanceof MethodInvocation) {
          MethodInvocation m = (MethodInvocation) parent;
          for (final Object arg : m.arguments()) {
            Rename("MethodArgument", variableRewrites, (ASTNode) node);
          }
          Rename("MethodInvocation", methodRewrites, node);
        } else if (parent instanceof SimpleType) {
          Rename("SimpleType", typeRewrites, node);
        } else if (parent instanceof InfixExpression) {
          Rename("InfixExpression", variableRewrites, node);
        } else if (parent instanceof PostfixExpression) {
          Rename("PostfixExpression", variableRewrites, node);
        } else if (parent instanceof PrefixExpression) {
          Rename("PrefixExpression", variableRewrites, node);
        } else if (parent instanceof ReturnStatement) {
          Rename("ReturnStatement", variableRewrites, node);
        } else if (parent instanceof ArrayAccess) {
          Rename("ArrayAccess", variableRewrites, node);
        } else if (parent instanceof QualifiedName) {
          Rename("QualifiedName", variableRewrites, node);
        } else if (parent instanceof IfStatement) {
          Rename("IfStatement", variableRewrites, node);
        } else if (parent instanceof ArrayCreation) {
          Rename("ArrayCreation", variableRewrites, node);
        } else if (parent instanceof Assignment) {
          Rename("Assignment", variableRewrites, node);
        } else if (parent instanceof MethodDeclaration ||
            parent instanceof VariableDeclarationFragment ||
            parent instanceof SingleVariableDeclaration ||
            parent instanceof TypeDeclaration) {
          // These have already been re-written.
        } else {
          System.err.println(
              "Unknown type " + parent.getClass().getName() + " for name "
                  + node.toString());
        }

        return true;
      }
    });

    edits.add(this.traversalRewrite.rewriteAST(document, null));
  }
}
