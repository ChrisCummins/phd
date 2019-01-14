# Blink

The blink program flashes an LED on and off. Hooray! The value of this is as
an exercise in developing Arduino code in a structured rigurous fashion.

**The program:** The program is defined as a templated class in teh file 
`blink.h`. The template allows it to use either a concrete arduino interface
(for running on hardware) or a mock interface class.

**Testing locally:** Because we define our program using the interface 
abstraction, we can use the mock class to run various tests, e.g. ensuring that
the correct pin is set as an output, etc. These can be executed using:

```
$ bazel test //maker/arduino/blink:blink_test
```

**Deploying to hardware:** The file `arduino_uno_program.cc` unites the 
interface of the program with the concrete arduino interface. To compile and
upload the firmware to an attached Arduino device, run:

```
$ bazel run //maker/arduino/blink:arduino_uno_program
```
