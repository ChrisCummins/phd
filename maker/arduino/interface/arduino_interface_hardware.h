// Implementation of ArduinoInterface that uses the real Arduino hardware.
#pragma once

#include <stdint.h>

#include <Arduino_interface.h>

namespace arduino {

// Implements the Arduino interface using the real Arduino hardware layer.
class ArduinoImpl : public ArduinoInterface {
public:
  ~ArduinoImpl() override;

  virtual DigitalValue DigitalRead(uint8_t pin) const override;
  virtual void DigitalWrite(uint8_t pin, const DigitalValue& value) const override;
  virtual unsigned long Millis(void) const override;
  virtual void Delay(unsigned long ms) const override;
  virtual void SetPinMode(uint8_t pin, const PinMode& mode) const override;
};

} // namespace arduino
