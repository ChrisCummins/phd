// Driver for Java methods.
//
// Authors: Tingda Du, Chris Cummins.
//
// Copyright (c) 2019 Chris Cummins.
//
// DeepSmith is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DeepSmith is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
package deeplearning.deepsmith.harnesses;

import com.google.common.io.ByteStreams;
import deeplearning.clgen.preprocessors.JavaPreprocessor;
import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;

public class JavaDriver {

  // The mode used to create argument values.
  enum Mode {
    DEFAULT,
    RANDOM
  }

  /** The driver configuration. Specifies the rules used to produce inputs, and their size. */
  public static class JavaDriverConfiguration {
    public JavaDriverConfiguration(final Mode mode, final int arrayLength, final long seed) {
      this.mode = mode;
      this.arrayLength = arrayLength;
      this.seed = seed;
    }

    public final Mode mode;
    public int arrayLength;
    public long seed;
  }

  /** The result of driving a Java Method. */
  public static class JavaDriverResult {
    public JavaDriverResult() {
      succeeded_ = false;
      failed_ = false;
      mutableParameterValues_ = new ArrayList<Object>();
    }

    public int getMutableParameterCount() {
      return mutableParameterValues_.size();
    }

    public String getMutableParameterValue(int i) {
      return String.format("%s", mutableParameterValues_.get(i));
    }

    public String getReturnValue() {
      return String.format("%s", returnValue_);
    }

    public String toString() {
      if (!succeeded_ && !failed_) {
        return "Unknown driver result";
      } else if (!succeeded_) {
        assert failed_;
        return errorMessage_;
      }

      StringBuilder builder = new StringBuilder();
      for (int i = 0; i < getMutableParameterCount(); ++i) {
        builder.append(String.format("Par[%d]: %s\n", i, getMutableParameterValue(i)));
      }
      builder.append(String.format("Return: %s", getReturnValue()));
      return builder.toString();
    }

    public void SetSuccess() {
      assert !failed_;
      assert !succeeded_;
      succeeded_ = true;
    }

    // Don't call this directly, use SetInvalidDriverInput() or
    // SetValidDriverFailure() instead.
    private void SetFailed() {
      assert !failed_;
      assert !succeeded_;
      failed_ = true;
    }

    public void SetValidDriverFailure(final String message) {
      errorMessage_ = message;
      SetFailed();
    }

    public void SetReturnValue(final Object object) {
      returnValue_ = object;
    }

    public void AddMutableParameterValue(final Object object) {
      mutableParameterValues_.add(object);
    }

    public void SetInvalidDriverInput(final String message) {
      errorMessage_ = message;
      SetFailed();
    }

    public void SetInternalDriverFailure(final String message) {
      errorMessage_ = message;
      SetFailed();
    }

    public boolean IsFailure() {
      return failed_;
    }

    public boolean IsSuccess() {
      return succeeded_;
    }

    // We need to track both success and failure as there is a third state,
    // uninitialised.
    private boolean succeeded_;
    private boolean failed_;
    private String errorMessage_;
    private Object returnValue_;
    private ArrayList<Object> mutableParameterValues_;
  }

  /** Abstract base class for errors during driving. */
  public abstract static class DriverException extends Exception {
    public DriverException(String message) {
      super(message);
    }
  }

  /**
   * A "something went wrong" error with the driver. This should not occur in normal use and
   * indicate a bug in the driver.
   */
  public static class InternalDriverException extends DriverException {
    public InternalDriverException(String message) {
      super(message);
    }
  }

  /**
   * Exception raised when an invalid input is passed to the driver. E.g. a string which does not
   * compile, or an unsupported parameter type.
   */
  public static class InvalidDriverInputException extends DriverException {
    public InvalidDriverInputException(String message) {
      super(message);
    }
  }

  /** A "valid" error during method execution. I.e. the called method raises an exception. */
  public static class ValidDriverInputException extends DriverException {
    public ValidDriverInputException(String message) {
      super(message);
    }
  }

  /** An in-memory java source file. */
  public class JavaSourceFromString extends SimpleJavaFileObject {
    final String code_;

    public JavaSourceFromString(String name, String code) {
      // Replace package qualifiers with directory separators when building the
      // name.
      super(URI.create("string:///" + name.replace('.', '/') + Kind.SOURCE.extension), Kind.SOURCE);
      code_ = code;
    }

    @Override
    public CharSequence getCharContent(boolean unusedIgnoreEncodingErrors) {
      return code_;
    }
  }

  /** The factory class for producing values of given types. */
  public static class ValueGenerator {
    public ValueGenerator(final JavaDriverConfiguration config) {
      config_ = config;
      rng_ = new Random();
      rng_.setSeed(config_.seed);
    }

    /**
     * Generate a value of the given type.
     *
     * @param parameterType The type of value to produce.
     * @return A value of the given type. The value is determined by the JavaDriverConfiguration
     *     settings.
     * @throws InvalidDriverInputException If a value of the given type cannot be produced.
     */
    public Object Generate(Class<?> type) throws InvalidDriverInputException {
      if (type.isPrimitive()) {
        return GeneratePrimitive(type);
      } else if (type.isArray()) {
        return GenerateArray(type.getComponentType());
      } else {
        // For non-primitive and non-arrays, try using the default constructor.
        try {
          return type.getConstructor().newInstance();
        } catch (NoSuchMethodException e) {
          throw new InvalidDriverInputException(
              String.format("Unable to construct parameter of type `%s`", type.getSimpleName()));
        } catch (InvocationTargetException e) {
          throw new InvalidDriverInputException(
              String.format(
                  "Constructor for type `%s` raised exception `%s`", type.getSimpleName(), e));
        } catch (InstantiationException e) {
          throw new InvalidDriverInputException(
              String.format(
                  "Constructor for type `%s` raised exception `%s`", type.getSimpleName(), e));
        } catch (IllegalAccessException e) {
          throw new InvalidDriverInputException(
              String.format("Constructor for type `%s` cannot be accessed", type.getSimpleName()));
        }
      }
    }

    // Private generator methods.

    /**
     * Generate an array of values of the given type.
     *
     * @param elementType The type of the element to produce.
     */
    private Object GenerateArray(Class<?> elementType) throws InvalidDriverInputException {
      final int arrayLength = config_.arrayLength;
      Object arrayDefaults = Array.newInstance(elementType, arrayLength);
      for (int i = 0; i < arrayLength; ++i) {
        Array.set(arrayDefaults, i, Generate(elementType));
      }
      return arrayDefaults;
    }

    /**
     * Produce a primitive of the given type.
     *
     * @param type The type to generate.
     * @return An object of the given type.
     * @throws InvalidDriverInputException If the type is not supported.
     */
    private Object GeneratePrimitive(Class<?> type) throws InvalidDriverInputException {
      if (type == Boolean.TYPE) {
        return GenerateBoolean();
      } else if (type == Character.TYPE) {
        return GenerateCharacter();
      } else if (type == Byte.TYPE) {
        return GenerateByte();
      } else if (type == Short.TYPE) {
        return GenerateShort();
      } else if (type == Integer.TYPE) {
        return GenerateInteger();
      } else if (type == Long.TYPE) {
        return GenerateLong();
      } else if (type == Float.TYPE) {
        return GenerateFloat();
      } else if (type == Double.TYPE) {
        return GenerateDouble();
      }
      throw new InvalidDriverInputException("Unsupport primitive type");
    }

    private Object GenerateBoolean() {
      if (config_.mode == Mode.DEFAULT) {
        return true;
      } else {
        return rng_.nextBoolean();
      }
    }

    private Object GenerateCharacter() {
      if (config_.mode == Mode.DEFAULT) {
        return 'A';
      } else {
        return (char) (rng_.nextInt(95) + 32);
      }
    }

    private Object GenerateByte() {
      if (config_.mode == Mode.DEFAULT) {
        return 0;
      } else {
        byte[] value = new byte[1];
        rng_.nextBytes(value);
        return value[0];
      }
    }

    private Object GenerateShort() {
      if (config_.mode == Mode.DEFAULT) {
        return config_.arrayLength;
      } else {
        return (short) (rng_.nextInt(65536) - 32768);
      }
    }

    private Object GenerateInteger() {
      if (config_.mode == Mode.DEFAULT) {
        return config_.arrayLength;
      } else {
        return rng_.nextInt();
      }
    }

    private Object GenerateLong() {
      if (config_.mode == Mode.DEFAULT) {
        return config_.arrayLength;
      } else {
        return rng_.nextLong();
      }
    }

    private Object GenerateFloat() {
      if (config_.mode == Mode.DEFAULT) {
        return 0;
      } else {
        return Float.MIN_VALUE + (Float.MAX_VALUE - Float.MIN_VALUE) * rng_.nextFloat();
      }
    }

    private Object GenerateDouble() {
      if (config_.mode == Mode.DEFAULT) {
        return 0;
      } else {
        return Double.MIN_VALUE + (Double.MAX_VALUE - Double.MIN_VALUE) * rng_.nextDouble();
      }
    }

    private final JavaDriverConfiguration config_;
    private Random rng_;
  }

  public JavaDriver(final JavaDriverConfiguration config) {
    preprocessor_ = new JavaPreprocessor();
    compiler_ = ToolProvider.getSystemJavaCompiler();
    config_ = config;
  }

  /**
   * Create a temporary directory with an informative prefix. Crashes on error.
   *
   * <p>It is the responsibility of the calling code to delete this directory once finished. Call
   * JavaDriver.DeleteTree().
   *
   * @return The path of the temporary directory.
   */
  public static Path CreateTemporaryDirectoryOrDie() {
    try {
      return Files.createTempDirectory("phd_deeplearning_deepsmith_harnesses_JavaDriver_");
    } catch (IOException e) {
      System.err.println("[driver error] Failed to create temporary directory");
      System.exit(1);
      return null;
    }
  }

  /**
   * Delete a directory containing zero or more files.
   *
   * <p>This is not recursive, only files are allowed.
   *
   * @param path The directory to delete.
   */
  public static void DeleteTree(final File path) {
    String[] entries = path.list();
    for (String s : entries) {
      File currentFile = new File(path.getPath(), s);
      currentFile.delete();
    }
    // Now delete the empty directory.
    path.delete();
  }

  private static Class<?> LoadClassFromFile(final File classDirectory, final String className)
      throws InternalDriverException {
    try {
      URL url = classDirectory.toURI().toURL();
      URLClassLoader classLoader = URLClassLoader.newInstance(new URL[] {url});
      return classLoader.loadClass(className);
    } catch (MalformedURLException e) {
      throw new InternalDriverException("Failed to create URL");
    } catch (ClassNotFoundException e) {
      throw new InternalDriverException("Failed to load compiled class");
    }
  }

  public Object[] CreateValuesForParameters(Class<?>[] parameterTypes)
      throws InvalidDriverInputException {
    ValueGenerator generator = new ValueGenerator(config_);

    Object[] parameters = new Object[parameterTypes.length];
    for (int i = 0; i < parameterTypes.length; ++i) {
      Class<?> parameterType = parameterTypes[i];
      Object value = generator.Generate(parameterType);

      // Array types must be cast, but scalar types cannot.
      if (parameterType.isArray()) {
        parameters[i] = parameterType.cast(value);
      } else {
        parameters[i] = value;
      }
    }
    return parameters;
  }

  public static Method GetMethodFromClass(Class<?> parentClass) throws InvalidDriverInputException {
    Method[] methods = parentClass.getDeclaredMethods();
    if (methods.length == 0) {
      throw new InvalidDriverInputException("Class contains no methods, need one");
    } else if (methods.length > 1) {
      throw new InvalidDriverInputException(
          String.format("Class contains %d methods, need one", methods.length));
    }
    return methods[0];
  }

  // Driver methods.

  public JavaDriverResult Drive(final Path workingDirectory, final String methodSrc) {
    final String classSrc = preprocessor_.WrapMethodInClassWithShim(methodSrc);
    final JavaSourceFromString javaSrc = new JavaSourceFromString("A", classSrc);
    return Drive(workingDirectory, javaSrc);
  }

  public JavaDriverResult Drive(final Path workingDirectory, final JavaFileObject javaSrc) {
    Iterable<? extends JavaFileObject> fileObjects = Arrays.asList(javaSrc);

    List<String> options = new ArrayList<String>();
    options.add("-d");
    options.add(workingDirectory.toString());

    StringWriter output = new StringWriter();
    boolean success =
        compiler_
            .getTask(
                /*out=*/ output,
                /*fileManager=*/ null,
                /*diagnosticListener=*/ null,
                /*options=*/ options,
                /*classes=*/ null,
                /*compilationUnits=*/ fileObjects)
            .call();

    if (!success) {
      JavaDriverResult result = new JavaDriverResult();
      result.SetInvalidDriverInput("Failed to compile class: " + output);
      return result;
    }

    String className = javaSrc.getName();
    // Strip trailing extension.
    className = className.replaceFirst("[.][^.]+$", "");
    // Strip leading path.
    className = className.replaceFirst("^.*/", "");

    try {
      Class<?> parentClass = LoadClassFromFile(workingDirectory.toFile(), className);
      Method method = GetMethodFromClass(parentClass);
      return Drive(method);
    } catch (InternalDriverException e) {
      JavaDriverResult result = new JavaDriverResult();
      result.SetInternalDriverFailure(e.getMessage());
      return result;
    } catch (InvalidDriverInputException e) {
      JavaDriverResult result = new JavaDriverResult();
      result.SetInvalidDriverInput(e.getMessage());
      return result;
    }
  }

  /**
   * Drive the given Java method.
   *
   * <p>This method will attempt to catch all errors and return a JavaDriverResult, but I haven't
   * done an exhaustive check, and there may still remain room for things to break.
   *
   * @param method The method to drive.
   * @return A JavaDriverResult instance.
   */
  public JavaDriverResult Drive(final Method method) {
    JavaDriverResult result = new JavaDriverResult();
    Class<?>[] parameterTypes = method.getParameterTypes();

    try {
      Object[] parameters = CreateValuesForParameters(parameterTypes);
      Class<?> parentClass = method.getDeclaringClass();
      Object instance = parentClass.newInstance();
      result.SetSuccess();
      result.SetReturnValue(method.invoke(instance, parameters));

      for (int i = 0; i < parameters.length; ++i) {
        Class<?> parameterType = parameterTypes[i];
        Object parameter = parameters[i];
        boolean isFinal = ((parameterType.getModifiers() & Modifier.FINAL) == Modifier.FINAL);

        if (!isFinal) {
          result.AddMutableParameterValue(parameter);
        }
      }
    } catch (InvalidDriverInputException e) {
      result.SetInvalidDriverInput(e.getMessage());
    } catch (InstantiationException e) {
      result.SetInvalidDriverInput("Class cannot be instantiated");
    } catch (IllegalAccessException e) {
      result.SetInvalidDriverInput("Method is inaccessible");
    } catch (InvocationTargetException e) {
      result.SetValidDriverFailure("Method threw exception: " + e);
    }

    return result;
  }

  private JavaPreprocessor preprocessor_;
  private JavaCompiler compiler_;
  private final JavaDriverConfiguration config_;

  public static void main(String[] args) throws Throwable {
    Mode mode = Mode.DEFAULT;
    int arrayLength = 10;
    long seed = 0;
    JavaDriverConfiguration config = new JavaDriverConfiguration(mode, arrayLength, seed);

    JavaDriver javaDriver = new JavaDriver(config);

    final String input = new String(ByteStreams.toByteArray(System.in));
    final Path tmpDir = JavaDriver.CreateTemporaryDirectoryOrDie();

    try {
      JavaDriverResult result = javaDriver.Drive(tmpDir, input);
      System.out.println(result.toString());
    } finally {
      JavaDriver.DeleteTree(tmpDir.toFile());
    }
  }
}
