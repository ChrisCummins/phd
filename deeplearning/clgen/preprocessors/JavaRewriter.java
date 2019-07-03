// A Java AST rewriter.
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
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.ToolFactory;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.EnumConstantDeclaration;
import org.eclipse.jdt.core.dom.EnumDeclaration;
import org.eclipse.jdt.core.dom.IBinding;
import org.eclipse.jdt.core.dom.MarkerAnnotation;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.NormalAnnotation;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.SingleVariableDeclaration;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.formatter.CodeFormatter;
import org.eclipse.jdt.internal.compiler.impl.CompilerOptions;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.MalformedTreeException;
import org.eclipse.text.edits.TextEdit;
import org.eclipse.text.edits.UndoEdit;

/** A rewriter for Java source code. */
public class JavaRewriter {

  private static final String[] RESERVED_WORDS_ =
      new String[] {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "true",
        "try",
        "void",
        "volatile",
        "while"
      };

  private static final Set<String> RESERVED_WORDS = new HashSet<>(Arrays.asList(RESERVED_WORDS_));
  private ASTRewrite traversalRewrite;
  private AST traversalAST; // Abstract Syntax Tree
  private HashMap<String, String> methodRewrites = new HashMap<>();
  private HashMap<String, String> typeRewrites = new HashMap<>();
  private HashMap<String, String> variableRewrites = new HashMap<>();
  private HashMap<String, String> labelRewrites = new HashMap<>();
  private HashMap<IBinding, String> bindingsRewrites = new HashMap<>();

  /**
   * Strip the comments from a compilation unit.
   *
   * <p>This iterates *backwards* over the comments, creating a DeleteEdit instance for each
   * comment. These edits must be applied in the order they are returned, since we use simple
   * indexing into the string source to determine where to snip the code.
   *
   * @param edits The edit list to append comment-removing edits to.
   * @param compilationUnit The compilation unit to remove comments from.
   * @deprecated StripComments() is no longer required, because the pre-processing pipeline has been
   *     modified so as to delte comments. See:
   *     //datasets/github/scrape_repos/preprocessors:extractors.
   */
  private static void StripComments(
      ArrayList<TextEdit> edits, final CompilationUnit compilationUnit) {

    final List<Comment> comments = compilationUnit.getCommentList();
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
    codeFormatterProperties.setProperty(JavaCore.COMPILER_SOURCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(JavaCore.COMPILER_COMPLIANCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, CompilerOptions.VERSION_1_8);
    // Allow really, really long lines.
    codeFormatterProperties.setProperty("org.eclipse.jdt.core.formatter.lineSplit", "9999");
    codeFormatterProperties.setProperty(
        "org.eclipse.jdt.core.formatter.comment.line_length", "9999");
    CodeFormatter codeFormatter = ToolFactory.createCodeFormatter(codeFormatterProperties);
    edits.add(
        codeFormatter.format(
            CodeFormatter.K_COMPILATION_UNIT, source, 0, source.length(), 0, null));
    ApplyEdits(edits, document);
    return document.get();
  }

  /**
   * Get the compilation unit for a document.
   *
   * @param document The document to get the compilation unit for.
   * @param filename The name of the file.
   * @return The compilation unit.
   */
  private static CompilationUnit GetCompilationUnit(
      final Document document, final String filename) {
    ASTParser parser = ASTParser.newParser(AST.JLS8);
    parser.setSource(document.get().toCharArray());
    parser.setResolveBindings(true);
    parser.setKind(ASTParser.K_COMPILATION_UNIT);
    parser.setBindingsRecovery(true);

    parser.setCompilerOptions(JavaCore.getOptions());
    parser.setUnitName(filename);

    parser.setEnvironment(
        /*classpathEntries=*/ null,
        /*sourcepathEntries=*/ null,
        /*encodings=*/ null,
        /*includeRunningVMBootclasspath=*/ true);

    return (CompilationUnit) parser.createAST(null);
  }

  /**
   * Apply a list of edits to a document.
   *
   * @param edits The edits to apply.
   * @param document The document to apply the edits to.
   */
  private static void ApplyEdits(final List<TextEdit> edits, Document document) {
    for (TextEdit edit : edits) {
      try {
        UndoEdit undo = edit.apply(document); // Do something if it fails. Fall back?
      } catch (MalformedTreeException e) {
        e.printStackTrace();
      } catch (BadLocationException e) {
        e.printStackTrace();
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
  private static String ReadFile(String path, Charset encoding) throws IOException {
    byte[] encoded = Files.readAllBytes(Paths.get(path));
    return new String(encoded, encoding);
  }

  /**
   * Write a file and return its contents as a string.
   *
   * @param path The path of the file to write.
   * @param encoding The encoding of the file.
   * @param Contents to be written to the file.
   * @return True if Writing is successful, False otherwise (throws exception)
   * @throws IOException In case of IO error.
   */
  private static boolean WriteFile(String path, Charset encoding, String corpus)
      throws IOException {
    byte[] bytes = corpus.getBytes();
    Files.write(
        Paths.get(path), bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    return true;
  }

  /**
   * @param input file name
   * @param command line arguments
   * @return True if input parsing is successful, False otherwise
   * @throws IOException
   */
  public boolean processInput(final String fileName, final String[] args) throws IOException {

    /*
     * There's a big problem where if the code contains integers in the form of 123_456_789
     * Compilation Unit won't be able to produce an AST. This is only one case I found, there
     * might be other cases like that.
     */

    String extension = fileName.substring(fileName.lastIndexOf(".") + 1);
    if (!extension.equals("java")) {
      throw new IOException("File input must be a java program");
    }

    final String javaCode = ReadFile(fileName, Charset.defaultCharset());

    final String source = RewriteSource(javaCode, fileName);
    if (source == null) {
      System.err.println("fatal: RewriteSource() returned null.");
      System.exit(1);
    }
    System.out.println(source);

    String className = fileName.substring(0, fileName.lastIndexOf("."));
    String newClassName = this.typeRewrites.get(className);
    assert (newClassName != null);
    return WriteFile(newClassName + ".java", Charset.defaultCharset(), source);
  }

  public static void main(final String[] args) {
    JavaRewriter rewriter = new JavaRewriter();

    try {
      final String input = new String(ByteStreams.toByteArray(System.in));
      final String source = rewriter.RewriteSource(input, "A.java");
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
   * @param binding The binding of the thing to be rewritten.
   * @param name The current name.
   * @param base_char The base character to use for generating new names, e.g. 'A' to produce the
   *     sequence 'A', 'B', 'C'...
   * @param name_prefix An optional prefix to prepend to generated names.
   * @return The replacement name.
   */
  private String GetNextName(
      HashMap<String, String> rewrites,
      IBinding binding,
      final String name,
      final char base_char,
      final String name_prefix) {
    StringBuilder s = new StringBuilder();
    int i = rewrites.size();

    assert (i == bindingsRewrites.size());

    s.append(name_prefix);
    // Build the new name character by character
    while (i > 25) {
      int k = i / 26;
      i %= 26;
      s.append((char) (base_char - 1 + k));
    }
    s.append((char) (base_char + i));

    String newName = s.toString();

    // Check that it isn't a reserved word, or else generate a new one.
    if (RESERVED_WORDS.contains(newName)) {
      // Insert a "dummy" value using an illegal identifier name.
      s.append("\t!@invalid@!\t");
      s.append(bindingsRewrites.size());
      final String invalidIdentifier = s.toString();
      rewrites.put(invalidIdentifier, invalidIdentifier);
      return GetNextName(rewrites, binding, name, base_char, name_prefix);
    }

    if (bindingsRewrites.containsKey(binding)) {
      newName = bindingsRewrites.get(binding);
    } else if (binding != null) {
      bindingsRewrites.put(binding, newName);
      if (rewrites.containsKey(name)) {
        Random rand = new Random();
        String randName = "";
        while (rewrites.containsKey(randName)) {
          randName = name + rand.nextInt(100_000);
        }
        rewrites.put(randName, newName);
      } else {
        rewrites.put(name, newName);
      }
    }
    return newName;
  }

  /**
   * Rewrite a Java source file to make it more amenable to machine learning.
   *
   * @param source The source code to rewrite.
   * @param filename The name of the file to rewrite.
   * @return The rewritten source code.
   */
  public String RewriteSource(String source, final String filename) {
    Document document = new Document(source);
    CompilationUnit compilationUnit = GetCompilationUnit(document, filename);

    ArrayList<TextEdit> edits = new ArrayList<>();
    RewriteIdentifiers(edits, compilationUnit, document);
    ApplyEdits(edits, document);
    // Format the source code.
    return FormatSource(document.get());
  }

  private void RewriteIdentifiers(
      ArrayList<TextEdit> edits, final CompilationUnit compilationUnit, final Document document) {
    this.traversalRewrite = ASTRewrite.create(compilationUnit.getAST());
    this.traversalAST = compilationUnit.getAST();

    // Rewrite declarations.

    compilationUnit.accept(
        new ASTVisitor() {
          /* Private helper method for renaming declarations. */
          private boolean Rename(
              final String type_name,
              final HashMap<String, String> rewriteTable,
              IBinding binding,
              final ASTNode node,
              final char base_char,
              final String name_prefix) {
            final String oldName = node.toString();
            final String newName =
                GetNextName(rewriteTable, binding, oldName, base_char, name_prefix);
            traversalRewrite.replace(node, traversalAST.newSimpleName(newName), null);
            return true;
          }

          @Override
          public boolean visit(NormalAnnotation node) {
            traversalRewrite.remove(node, null);
            return super.visit(node);
          }

          // Annotations such as @Override
          @Override
          public boolean visit(MarkerAnnotation node) {
            traversalRewrite.remove(node, null);
            return super.visit(node);
          }

          @Override
          public boolean visit(TypeDeclaration node) {
            return Rename(
                "TypeDeclaration", typeRewrites, node.resolveBinding(), node.getName(), 'A', "");
          }

          @Override
          public boolean visit(MethodDeclaration node) {
            if (!node.getName().toString().equals("main")
                && !node.getName().toString().equals("run")
                && !node.isConstructor()) {
              Rename(
                  "MethodDeclaration",
                  methodRewrites,
                  node.resolveBinding(),
                  node.getName(),
                  'A',
                  "fn_");
            } else if (node.isConstructor()) {
              Rename(
                  "MethodDeclaration",
                  typeRewrites,
                  node.resolveBinding(),
                  node.getName(),
                  'A',
                  "");
            }
            return true;
          }

          @Override
          public boolean visit(SingleVariableDeclaration node) {
            return Rename(
                "SingleVariableDeclaration",
                variableRewrites,
                node.resolveBinding(),
                node.getName(),
                'a',
                "");
          }

          @Override
          public boolean visit(VariableDeclarationFragment node) {
            return Rename(
                "VariableDeclarationFragment",
                variableRewrites,
                node.resolveBinding(),
                node.getName(),
                'a',
                "");
          }

          @Override
          public boolean visit(EnumDeclaration node) {
            return Rename(
                "EnumDeclaration",
                variableRewrites,
                node.resolveBinding(),
                node.getName(),
                'A',
                "");
          }

          @Override
          public boolean visit(EnumConstantDeclaration node) {
            return Rename(
                "EnumConstantDeclaration",
                variableRewrites,
                node.resolveConstructorBinding(),
                node.getName(),
                'A',
                "");
          }

          @Override
          public boolean visit(SimpleName node) {
            final ASTNode parent = node.getParent();

            switch (parent.getNodeType()) {
              case ASTNode.CONTINUE_STATEMENT:
                return Rename(
                    "ContinueStatement", labelRewrites, node.resolveBinding(), node, 'a', "");
              case ASTNode.LABELED_STATEMENT:
                return Rename(
                    "LabeledStatement", labelRewrites, node.resolveBinding(), node, 'a', "");
              case ASTNode.BREAK_STATEMENT:
                return Rename(
                    "BreakStatement", labelRewrites, node.resolveBinding(), node, 'a', "");
              case ASTNode.MARKER_ANNOTATION:
              default:
                break;
            }
            return super.visit(node);
          }
        });

    // Rewrite usages.
    compilationUnit.accept(
        new ASTVisitor() {

          /* Private helper method to rename usages. */
          private boolean Rename(final String type_name, final ASTNode node, IBinding binding) {
            if (!bindingsRewrites.containsKey(binding)) {
              return false;
            }

            final String newName = bindingsRewrites.get(binding);
            traversalRewrite.replace(node, traversalAST.newSimpleName(newName), null);
            return true;
          }

          @Override
          public boolean visit(SimpleName node) {
            final ASTNode parent = node.getParent();
            boolean success = false;
            IBinding binding = node.resolveBinding();

            switch (parent.getNodeType()) {
              case ASTNode.METHOD_INVOCATION:
                success = Rename("MethodInvocation", node, binding);
                if (!success) Rename("MethodInvocation", node, binding);
                break;
              case ASTNode.SIMPLE_TYPE:
                Rename("SimpleType", node, binding);
                break;
              case ASTNode.INFIX_EXPRESSION:
                Rename("InfixExpression", node, binding);
                break;
              case ASTNode.POSTFIX_EXPRESSION:
                Rename("PostfixExpression", node, binding);
                break;
              case ASTNode.PREFIX_EXPRESSION:
                Rename("PrefixExpression", node, binding);
                break;
              case ASTNode.RETURN_STATEMENT:
                Rename("ReturnStatement", node, binding);
                break;
              case ASTNode.ARRAY_ACCESS:
                Rename("ArrayAccess", node, binding);
                break;
              case ASTNode.QUALIFIED_NAME:
                if (!node.toString().matches("length")) {
                  Rename("QualifiedName", node, binding);
                }
                break;
              case ASTNode.IF_STATEMENT:
                Rename("IfStatement", node, binding);
                break;
              case ASTNode.ARRAY_CREATION:
                Rename("ArrayCreation", node, binding);
                break;
              case ASTNode.ASSIGNMENT:
                Rename("Assignment", node, binding);
                break;
              case ASTNode.MARKER_ANNOTATION:
                Rename("MarkerAnnotation", node, binding);
                break;
              case ASTNode.CLASS_INSTANCE_CREATION:
                Rename("ClassInstanceCreation", node, binding);
                break;
              case ASTNode.FIELD_ACCESS:
                Rename("FieldAccess", node, binding);
                break;
              case ASTNode.CONTINUE_STATEMENT:
                Rename("ContinueStatement", node, binding);
                break;
              case ASTNode.LABELED_STATEMENT:
                Rename("LabeledStatement", node, binding);
                break;
              case ASTNode.BREAK_STATEMENT:
                Rename("BreakStatement", node, binding);
                break;
              case ASTNode.CONDITIONAL_EXPRESSION:
                Rename("ConditionalExpression", node, binding);
                break;
              case ASTNode.VARIABLE_DECLARATION_FRAGMENT:
                Rename("VariableDeclarationFragment", node, binding);
                break;
              case ASTNode.SINGLE_VARIABLE_DECLARATION:
                Rename("SingleVariableDeclaration", node, binding);
                break;
              case ASTNode.CAST_EXPRESSION:
                Rename("CastExpression", node, binding);
                break;
              case ASTNode.ARRAY_INITIALIZER:
                Rename("ArrayInitializer", node, binding);
                break;
              case ASTNode.PARENTHESIZED_EXPRESSION:
                Rename("ParenthesizedExpression", node, binding);
                break;
              case ASTNode.INSTANCEOF_EXPRESSION:
                Rename("InstanceOfExpression", node, binding);
                break;
              case ASTNode.SWITCH_STATEMENT:
                Rename("SwitchStatement", node, binding);
                break;
              case ASTNode.SWITCH_CASE:
                Rename("SwitchCase", node, binding);
                break;
              case ASTNode.SYNCHRONIZED_STATEMENT:
                Rename("SynchronizedStatement", node, binding);
                break;
              case ASTNode.THROW_STATEMENT:
                Rename("ThrowStatement", node, binding);
                break;
              case ASTNode.WHILE_STATEMENT:
                Rename("WhileStatement", node, binding);
                break;
              case ASTNode.DO_STATEMENT:
                Rename("DoStatement", node, binding);
                break;
              case ASTNode.ASSERT_STATEMENT:
                Rename("AssertStatement", node, binding);
                break;
              case ASTNode.ENHANCED_FOR_STATEMENT:
                Rename("EnhancedForStatement", node, binding);
                break;
              case ASTNode.TRY_STATEMENT:
                Rename("TryStatement", node, binding);
                break;
                // case ASTNode.METHOD_DECLARATION:
                // case ASTNode.VARIABLE_DECLARATION_FRAGMENT:
                // case ASTNode.SINGLE_VARIABLE_DECLARATION:
                // case ASTNode.TYPE_DECLARATION:
              default:
                break;
            }

            return true;
          }
        });

    edits.add(this.traversalRewrite.rewriteAST(document, null));
  }
}
