#pragma once

#include <stdint.h>

namespace arduino {

// An analog (PWM wave) value.
class AnalogValue {
 public:
  // Constructors.
  AnalogValue(int value) : value_(value) {}
  static AnalogValue Max();
  static AnalogValue Min();

  // Explicit conversion to type used in Arduino API. Note that this casts a
  // 32 bit signed int to an 8 bit unsigned int.
  explicit operator uint8_t() const noexcept {
    return static_cast<uint8_t>(value_);
  }

  int ToInt() const { return static_cast<int>(value_); }

  bool operator<(const AnalogValue& rhs) const { return value_ < rhs.value_; }

  bool operator>(const AnalogValue& rhs) const { return value_ > rhs.value_; }

  bool operator==(const AnalogValue& rhs) const { return value_ == rhs.value_; }

  bool operator!=(const AnalogValue& rhs) const { return value_ != rhs.value_; }

  void operator-=(int offset) { value_ -= offset; }

  void operator+=(int offset) { value_ += offset; }

 private:
  int value_;
};

}  // namespace arduino
