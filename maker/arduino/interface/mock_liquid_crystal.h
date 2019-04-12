#pragma once

#include "gmock/gmock.h"

#include <stdint.h>

// A quick and hacky mock for a handful of the LiquidCrystal class functions I'm
// using.
// TODO(cec): Re-write this as a LiquidCrystalInterface, like with
// ArduinoInterface.
class LiquidCrystal {
 public:
  LiquidCrystal(uint8_t rs, uint8_t enable, uint8_t d0, uint8_t d1, uint8_t d2,
                uint8_t d3, uint8_t d4, uint8_t d5, uint8_t d6, uint8_t d7);
  LiquidCrystal(uint8_t rs, uint8_t rw, uint8_t enable, uint8_t d0, uint8_t d1,
                uint8_t d2, uint8_t d3, uint8_t d4, uint8_t d5, uint8_t d6,
                uint8_t d7);
  LiquidCrystal(uint8_t rs, uint8_t rw, uint8_t enable, uint8_t d0, uint8_t d1,
                uint8_t d2, uint8_t d3);
  LiquidCrystal(uint8_t rs, uint8_t enable, uint8_t d0, uint8_t d1, uint8_t d2,
                uint8_t d3);

  MOCK_CONST_METHOD0(clear, void());
  MOCK_CONST_METHOD1(print, void(const char*));
  MOCK_CONST_METHOD1(print, void(int));
  MOCK_CONST_METHOD2(begin, void(int, int));
  MOCK_CONST_METHOD2(setCursor, void(int, int));
};
