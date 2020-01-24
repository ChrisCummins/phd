# Blink

The blink program flashes an LED on and off. Hooray! The value of this is as
an exercise in developing Arduino code in a structured rigurous fashion.

**Files:**
* `blink.ino`: The main entrypoint for the Sketch. This is the code that will be
compiled and uploaded to hardware. However, this file doesn't implement the
business logic. It is merely a stub that calls into the logic defined in the
header.
* `blink.h`: Implements the business logic for the program, using a templated
class that allows it to be compiled for either real hardware or for mock
classes.
* `blink_test.cc`: C++ tests for the business logic, using the mock interface.
* `arduino_uno.cc`: A symlink back to the Sketch file. This is just to allow the
platform IO bazel rules to use this file as a `src`, since it won't accept
`.ino` file extensions.

**Testing locally:** Because we define our program using the interface
abstraction, we can use the mock class to run various tests, e.g. ensuring that
the correct pin is set as an output, etc. These can be executed using:

```
$ bazel test //maker/arduino/blink:blink_test
```

**Deploying to hardware:** The file `blink.ino` unites the
interface of the program with the concrete arduino interface. To compile and
upload the firmware to an attached Arduino Uno device, run:

```
$ bazel run //maker/arduino/blink:arduino_uno
```
