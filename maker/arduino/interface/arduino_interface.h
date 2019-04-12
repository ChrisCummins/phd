// A virtual interface over the functions exposed by Arduino.h.
//
// This is an extension of code released by Google. Below is the comment
// header for the initial release:
//
// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http:#www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

// TODO:
//#ifdef ARDUINO
//#define INTERFACE_HEADER(name) #include <Arduino_interface_ # name ".h"
//#else
//#define INTERFACE_HEADER(name) "maker/arduino/interface/" #name ".h"
//#endif  // ARDUINO

// If compiling for hardware, use the generated library header. For C++
// compilation, use the bazel header path.
#ifdef ARDUINO
#include <Arduino_interface_analog_value.h>
#include <Arduino_interface_digital_value.h>
#include <Arduino_interface_pin.h>
#include <Arduino_interface_pin_mode.h>
#else
#include "maker/arduino/interface/analog_value.h"
#include "maker/arduino/interface/digital_value.h"
#include "maker/arduino/interface/pin.h"
#include "maker/arduino/interface/pin_mode.h"
#endif

#include <stdint.h>

namespace arduino {

// Defines a C++ interface for accessing the Arduino hardware layer.
class ArduinoInterface {
 public:
  virtual ~ArduinoInterface() {}

  // Reads the value from a specified digital pin.
  // https://www.arduino.cc/en/Reference/DigitalRead
  virtual DigitalValue DigitalRead(const Pin& pin) const = 0;

  // Write a value to a digital pin.
  // https://www.arduino.cc/en/Reference/DigitalWrite
  virtual void DigitalWrite(const Pin& pin,
                            const DigitalValue& value) const = 0;

  // Returns the number of milliseconds since the Arduino board began running
  // the current program. This number will overflow (go back to zero), after
  // approximately 50 days.
  // https://www.arduino.cc/en/Reference/Millis
  virtual unsigned long Millis() const = 0;

  // Pauses the program for the amount of time (in milliseconds) specified as
  // parameter. (There are 1000 milliseconds in a second.)
  // https://www.arduino.cc/en/Reference/Delay
  virtual void Delay(unsigned long ms) const = 0;

  // Configures the specified pin to behave either as an input or an output.
  // https://www.arduino.cc/en/Reference/PinMode
  virtual void SetPinMode(const Pin& pin, const PinMode& mode) const = 0;

  // Sets the data rate in bits per second (baud) for serial data transmission.
  // https://www.arduino.cc/en/serial/begin
  virtual void InitSerial(unsigned long baud) const = 0;

  template <typename T>
  void SerialPrint(const T& value) const;

  // Set PWM range. Value must be in range [0,1023]. Wraps analogWriteRange().
  virtual void SetPwmRange(const int range) const = 0;

  // Writes an analog value (PWM wave) to a pin.
  // https://www.arduino.cc/reference/en/language/functions/analog-io/analogwrite/
  virtual void AnalogWrite(const Pin& pin, const AnalogValue& value) const = 0;

  // The pin number of the built in LED.
  static const Pin kBuiltInLedPin;
};

}  // namespace arduino
