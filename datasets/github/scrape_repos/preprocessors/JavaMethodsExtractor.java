/* Extract methods from Java source file and return a ListOfStrings proto.
 *
 * Usage:
 *     bazel run \
 *         //datasets/github/scrape_repos/preprocessors:JavaMethodsExtractor \
 *         < /path/to/file.java
 *
 * If environment variable $JAVA_METHOD_EXTRACTOR_STATIC_ONLY is set, only
 * static methods are returned.
 */
// Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package datasets.github.scrape_repos.preprocessors;

import com.google.common.io.ByteStreams;
import com.scrape_repos.ListOfStrings;
import java.io.IOException;
import java.lang.reflect.Modifier;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jface.text.Document;

/** Extract methods from Java source code. */
public class JavaMethodsExtractor {

  private ListOfStrings.Builder message;

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

  public static void main(final String[] args) {
    JavaMethodsExtractor extractor = new JavaMethodsExtractor();

    try {
      final String input = new String(ByteStreams.toByteArray(System.in));
      System.out.println(extractor.ExtractMethods(input).toString());
    } catch (IOException e) {
      System.err.println("fatal: I/O error");
      System.exit(1);
    }
  }

  /**
   * Return the string representation of a method declaration.
   *
   * <p>By default, a MethodDeclaration includes the JavaDoc comment. This strips that.
   *
   * @param method The method to stringify.
   * @returns The string source code of the method.
   */
  private String MethodDeclarationToString(MethodDeclaration method) {
    method.setJavadoc(null);
    return method.toString();
  }

  /**
   * Extract all methods from a Java source.
   *
   * @param source A Java source string.
   * @return A ListOfStrings proto instance.
   */
  private ListOfStrings ExtractMethods(String source) {
    Document document = new Document(source);
    message = ListOfStrings.newBuilder();

    final boolean staticOnly =
        !(System.getenv("JAVA_METHOD_EXTRACTOR_STATIC_ONLY") == null
            || System.getenv("JAVA_METHOD_EXTRACTOR_STATIC_ONLY").equals(""));

    try {
      CompilationUnit compilationUnit = GetCompilationUnit(document);

      if (staticOnly) {
        compilationUnit.accept(
            new ASTVisitor() {
              public boolean visit(MethodDeclaration node) {
                if ((node.getModifiers() & Modifier.STATIC) != 0) {
                  message.addString(MethodDeclarationToString(node));
                }
                return true;
              }
            });
      } else {
        compilationUnit.accept(
            new ASTVisitor() {
              public boolean visit(MethodDeclaration node) {
                message.addString(MethodDeclarationToString(node));
                return true;
              }
            });
      }
    } catch (IllegalArgumentException e) {
      System.err.println("error: Failed to parse unit.");
    }
    return message.build();
  }
}
