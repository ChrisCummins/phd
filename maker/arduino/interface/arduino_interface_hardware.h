// Implementation of ArduinoInterface that uses the real Arduino hardware.
#pragma once

#include <stdint.h>

#include <Arduino_interface.h>

namespace arduino {

// Implements the Arduino interface using the real Arduino hardware layer.
class ArduinoImpl : public ArduinoInterface {
public:
  ~ArduinoImpl() override;

  virtual DigitalValue DigitalRead(const Pin& pin) const override;
  virtual void DigitalWrite(const Pin& pin, const DigitalValue& value) const override;
  virtual unsigned long Millis(void) const override;
  virtual void Delay(unsigned long ms) const override;
  virtual void SetPinMode(const Pin& pin, const PinMode& mode) const override;
};

} // namespace arduino
