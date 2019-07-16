package deeplearning.clgen.preprocessors;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class JavaPreprocessorTest {

  @Test
  public void testCompilesWithoutErrorEmptyClass() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(pp.CompilesWithoutError("public class A {}"));
  }

  @Test
  public void testCompilesWithoutErrorEmptyFile() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(pp.CompilesWithoutError(""));
  }

  @Test
  public void testCompilesWithoutErrorWithEventListener() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(
        pp.CompilesWithoutError(
            "import java.util.EventListener;\n"
                + "public class A {\n"
                + "public static void Foo(EventListener a) {}}"));
  }

  @Test
  public void testCompilesWithoutErrorWithGeneric() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(
        pp.CompilesWithoutError(
            "import java.util.ArrayList;\n"
                + "public class A {\n"
                + "public static void Foo(ArrayList<Integer> a) {}}"));
  }

  @Test
  public void testCompilesWithoutErrorFail() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertFalse(pp.CompilesWithoutError("syntax error!"));
    assertFalse(pp.CompilesWithoutError("public static void Foo() {}"));

    // Class has wrong name.
    assertFalse(pp.CompilesWithoutError("public class B {}"));
  }
}
