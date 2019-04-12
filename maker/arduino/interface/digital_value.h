#pragma once

#include <stdint.h>

namespace arduino {

// A digital value. The return type of ArduinoInterface::DigitalRead(), and the
// second argument of ArduinoInterface::DigitalWrite().
class DigitalValue {
 public:
  // Constructors.
  static DigitalValue High();
  static DigitalValue Low();

  // Explicit conversion to type used in Arduino API.
  explicit operator uint8_t() const noexcept { return value_; }

 private:
  explicit DigitalValue(uint8_t value) : value_(value) {}

  uint8_t value_;
};

}  // namespace arduino
