// This file unites the blink implementation with the hardware Arduino
// interface. This is the code that will be executed on the microcontroller.

// Include the program implementation.
#include <Blink.h>

// Include the hardware implementation of the ArduinoInterface, and the
// base Arduino header. Even though the Arduino header is not used in this
// file, it must be #included for the build to succeed.
#include <Arduino.h>
#include <Arduino_hardware.h>

// Instantiate our concrete implementation.
arduino::blink::Blink<arduino::ArduinoImpl> program;

void setup() { program.Setup(); }

void loop() { program.Loop(); }
