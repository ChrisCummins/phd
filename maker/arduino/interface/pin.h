#pragma once

#include <stdint.h>

namespace arduino {

// A pin.
class Pin {
 public:
  explicit Pin(uint8_t pin) : pin_(pin) {}

  // Explicit conversion to the value expected by Arduino's pinMode() function.
  explicit operator uint8_t() const noexcept { return pin_; }

  // Enable equality checking between pins.
  bool operator==(const Pin& other) const { return other.pin_ == pin_; }

 private:
  uint8_t pin_;
};

}  // namespace arduino
