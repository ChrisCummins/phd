// An Arduino program which flashes an LED on and off.
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

// The pin number of the LED to flash.
static const uint8_t kLedToFlash = 13;

// The number of milliseconds to delay between flashing LED on and off.
static const int kDelayMilliseconds = 500;

// The program to execute.
template<typename ArduinoInterface>
class Blink {
 public:
  explicit Blink() : interface_() {}

  void Setup() {
    interface_.SetPinMode(kLedToFlash, PinMode::Output());
  }

  void Loop() {
    interface_.DigitalWrite(kLedToFlash, DigitalValue::High());
    interface_.Delay(kDelayMilliseconds);
    interface_.DigitalWrite(kLedToFlash, DigitalValue::Low());
    interface_.Delay(kDelayMilliseconds);
  }

  const ArduinoInterface& interface() const { return interface_; }

 private:
  const ArduinoInterface interface_;
};

}  // namespace blink
}  // namespace arduino
