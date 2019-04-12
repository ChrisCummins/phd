#pragma once

#include <stdint.h>

namespace arduino {

// A pin mode. This is the second argument to ArduinoInterface::SetPinMode().
class PinMode {
 public:
  // Constructors.
  static PinMode Output();
  static PinMode Input();
  static PinMode InputPullup();

  // Explicit conversion to the value expected by Arduino's pinMode() function.
  explicit operator uint8_t() const noexcept { return mode_; }

 private:
  explicit PinMode(uint8_t mode) : mode_(mode) {}

  uint8_t mode_;
};

}  // namespace arduino
