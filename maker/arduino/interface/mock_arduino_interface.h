#pragma once

#include "maker/arduino/interface/arduino_interface.h"
#include "maker/arduino/interface/mock_liquid_crystal.h"

#include "gmock/gmock.h"

#include <stdint.h>

namespace arduino {

// A mock of the ArduinoInterface.
class MockArduinoInterface : public ArduinoInterface {
 public:
  MOCK_CONST_METHOD1(DigitalRead, DigitalValue(const Pin& pin));
  MOCK_CONST_METHOD2(DigitalWrite,
                     void(const Pin& pin, const DigitalValue& value));
  MOCK_CONST_METHOD0(Millis, unsigned long());
  MOCK_CONST_METHOD1(Delay, void(unsigned long ms));
  MOCK_CONST_METHOD2(SetPinMode, void(const Pin& pin, const PinMode& mode));
  MOCK_CONST_METHOD1(InitSerial, void(unsigned long bud));
  MOCK_CONST_METHOD1(SetPwmRange, void(const int range));
  MOCK_CONST_METHOD2(AnalogWrite,
                     void(const Pin& pin, const AnalogValue& value));
};

}  // namespace arduino
