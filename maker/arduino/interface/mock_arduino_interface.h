#pragma once

#include "maker/arduino/interface/arduino_interface.h"

#include "gmock/gmock.h"

#include <stdint.h>

namespace arduino {

// A mock of the ArduinoInterface.
class MockArduinoInterface : public ArduinoInterface {
public:
  MOCK_CONST_METHOD1(DigitalRead, DigitalValue(uint8_t pin));
  MOCK_CONST_METHOD2(DigitalWrite, void(uint8_t pin, const DigitalValue& value));
  MOCK_CONST_METHOD0(Millis, unsigned long());
  MOCK_CONST_METHOD1(Delay, void(unsigned long ms));
  MOCK_CONST_METHOD2(SetPinMode, void(uint8_t pin, const PinMode& mode));
};

} // namespace arduino
