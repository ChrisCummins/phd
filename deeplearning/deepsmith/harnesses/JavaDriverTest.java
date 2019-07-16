package deeplearning.deepsmith.harnesses;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import deeplearning.deepsmith.harnesses.JavaDriver.Mode;
import java.lang.reflect.Method;
import java.nio.file.Path;
import junit.framework.TestCase;
import org.junit.Test;

public class JavaDriverTest extends TestCase {

  public JavaDriverTest() {
    defaultConfig_ = new JavaDriver.JavaDriverConfiguration(Mode.DEFAULT, 10, 0);
  }

  @Override
  protected void setUp() throws Exception {
    workingDir_ = JavaDriver.CreateTemporaryDirectoryOrDie();
  }

  @Override
  protected void tearDown() throws Exception {
    JavaDriver.DeleteTree(workingDir_.toFile());
  }

  public static class ReturnParamPlusFive {
    public int Method(int x) {
      return x + 5;
    }
  }

  @Test
  public void testDriveMethodThatReturnsParam() throws Exception {
    JavaDriver driver = new JavaDriver(defaultConfig_);

    Class<?> classToTest = ReturnParamPlusFive.class;
    Method methodToTest = classToTest.getMethods()[0];

    JavaDriver.JavaDriverResult result = driver.Drive(methodToTest);
    assertTrue(result.IsSuccess());
    assertEquals(result.getMutableParameterCount(), 0);
    // Value of input param is arrayLength.
    assertEquals(result.getReturnValue(), "15");
  }

  @Test
  public void testDriveEmptyString() throws Exception {
    JavaDriver driver = new JavaDriver(defaultConfig_);

    // An empty string will compile, but fail because it does not contain the
    // necessary single method.
    JavaDriver.JavaDriverResult result = driver.Drive(workingDir_, "");
    assertTrue(result.IsFailure());
    assertEquals(result.toString(), "Class contains no methods, need one");
  }

  @Test
  public void testDriveMultipleMethods() throws Exception {
    JavaDriver driver = new JavaDriver(defaultConfig_);

    // Only a single method can be driven at a time.
    JavaDriver.JavaDriverResult result =
        driver.Drive(workingDir_, "public void a() {} public void b() {}");
    assertTrue(result.IsFailure());
    assertEquals(result.toString(), "Class contains 2 methods, need one");
  }

  public class ACustomParamType {}

  public static class ParamWithInvalidType {
    public void Method(ACustomParamType x) {}
  }

  @Test
  public void testDriveParamWithInvalidType() throws Exception {
    JavaDriver driver = new JavaDriver(defaultConfig_);

    Class<?> classToTest = ParamWithInvalidType.class;
    Method methodToTest = classToTest.getMethods()[0];

    // An empty string will compile, but fail because it does not contain the
    // necessary single method.
    JavaDriver.JavaDriverResult result = driver.Drive(methodToTest);
    assertTrue(result.IsFailure());
    assertEquals("Unable to construct parameter of type `ACustomParamType`", result.toString());
  }

  private Path workingDir_;
  private JavaDriver.JavaDriverConfiguration defaultConfig_;
}
