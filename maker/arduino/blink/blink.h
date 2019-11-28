// An Arduino program which flashes the S-O-S sign in Morse Code.
#pragma once

// If compiling for hardware, use the generated library header. For C++
// compilation, use the bazel header path.
#ifdef ARDUINO
#include <Arduino_interface.h>
#else
#include "maker/arduino/interface/arduino_interface.h"
#endif

namespace arduino {
namespace blink {

// Set a pin high, weight for a number of milliseconds, then set the pin the low
// and weight for another number of milliseconds.
template <typename ArduinoInterface>
void HighDelayLow(const ArduinoInterface& interface, const Pin& pin,
                  const int high_duration_ms, const int low_duration_ms) {
  interface.DigitalWrite(pin, DigitalValue::High());
  interface.Delay(high_duration_ms);
  interface.DigitalWrite(pin, DigitalValue::Low());
  interface.Delay(low_duration_ms);
}

// The program to execute.
template <typename ArduinoInterface>
class Blink {
 public:
  explicit Blink() : interface_() {}

  void Setup() {
    interface_.SetPinMode(ArduinoInterface::kBuiltInLedPin, PinMode::Output());
  }

  void Loop() {
    // Disclaimer: I don't know morse code - this is likely very inaccurate.

    // dot-dot-dot
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150 + 75);
    // dash-dash-dash
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 200, 75);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 200, 75);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 200, 75 + 150);
    // dot-dot-dot
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150);
    HighDelayLow(interface_, ArduinoInterface::kBuiltInLedPin, 50, 150 + 600);
  }

  const ArduinoInterface& interface() const { return interface_; }

 private:
  const ArduinoInterface interface_;
};

}  // namespace blink
}  // namespace arduino
