#pragma once

#include <stdint.h>

namespace arduino {

// A 32 bit Internet Protocol v4 address.
class IPv4Address {
 public:
  // Constructor for dot-decimal notation.
  IPv4Address(uint8_t a, uint8_t b, uint8_t c, uint8_t d)
      : a_(a), b_(b), c_(c), d_(d) {}

  uint8_t a() const { return a_; }
  uint8_t b() const { return b_; }
  uint8_t c() const { return c_; }
  uint8_t d() const { return d_; }

 private:
  const uint8_t a_;
  const uint8_t b_;
  const uint8_t c_;
  const uint8_t d_;
};

}  // namespace arduino
