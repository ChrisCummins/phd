/* Extract methods from Java source file and return a ListOfStrings proto where
 * each string is the text source for a method.
 *
 * This process parses and rebuilds the AST. Formatting and comments (including
 * Javadoc) are lost.
 *
 * If environment variable $JAVA_METHOD_EXTRACTOR_STATIC_ONLY is set, only
 * static methods are returned.
 */
// Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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
import datasets.github.scrape_repos.ScrapeReposProtos.ListOfListOfStrings;
import datasets.github.scrape_repos.ScrapeReposProtos.ListOfStrings;
import java.io.IOException;
import java.lang.reflect.Modifier;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jface.text.Document;

/**
 * Extract methods from Java source code.
 */
public class JavaMethodsBatchedExtractor {

  private ListOfListOfStrings.Builder outer_message;
  private ListOfStrings.Builder inner_message;

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
    JavaMethodsBatchedExtractor extractor = new JavaMethodsBatchedExtractor();

    try {
      ListOfStrings input = ListOfStrings.parseFrom(
          ByteStreams.toByteArray(System.in));
      ListOfListOfStrings output = extractor.ProcessMessage(input);
      output.writeTo(System.out);
    } catch (IOException e) {
      System.err.println("fatal: I/O error");
      System.exit(1);
    }
  }

  /**
   * Return the string representation of a method declaration.
   *
   * By default, a MethodDeclaration includes the JavaDoc comment. This strips
   * that.
   *
   * @param method The method to stringify.
   * @returns The string source code of the method.
   */
  private String MethodDeclarationToString(MethodDeclaration method) {
    method.setJavadoc(null);
    return method.toString();
  }

  /**
   *
   * @param message
   */
  private ListOfListOfStrings ProcessMessage(ListOfStrings message) {
    final boolean staticOnly = !(
        System.getenv("JAVA_METHOD_EXTRACTOR_STATIC_ONLY") == null ||
            System.getenv("JAVA_METHOD_EXTRACTOR_STATIC_ONLY").equals(""));

    outer_message = ListOfListOfStrings.newBuilder();
    inner_message = ListOfStrings.newBuilder();

    for (int i = 0; i < message.getStringCount(); ++i) {
      final String src = message.getString(i);
      Document document = new Document(src);
      try {
        CompilationUnit compilationUnit = GetCompilationUnit(document);

        if (staticOnly) {
          compilationUnit.accept(new ASTVisitor() {
            public boolean visit(MethodDeclaration node) {
              if ((node.getModifiers() & Modifier.STATIC) != 0) {
                inner_message.addString(MethodDeclarationToString(node));
              }
              return true;
            }
          });
        } else {
          compilationUnit.accept(new ASTVisitor() {
            public boolean visit(MethodDeclaration node) {
              inner_message.addString(MethodDeclarationToString(node));
              return true;
            }
          });
        }
      } catch (IllegalArgumentException e) {
        System.err.println("error: Failed to parse unit.");
      }

      outer_message.addListOfStrings(inner_message);
      inner_message.clear();
    }

    return outer_message.build();
  }

}
