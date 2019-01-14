#include <Arduino_hardware.h>

#include <Arduino.h>
#include <Arduino_interface.h>

namespace arduino {

ArduinoImpl::~ArduinoImpl() {}

DigitalValue ArduinoImpl::DigitalRead(uint8_t pin) const {
  return digitalRead(pin) ? DigitalValue::High() : DigitalValue::Low();
}

void ArduinoImpl::DigitalWrite(uint8_t pin, const DigitalValue& value) const {
  return digitalWrite(pin, uint8_t(value));
}

unsigned long ArduinoImpl::Millis(void) const {
  return millis();
}

void ArduinoImpl::Delay(unsigned long ms) const {
  delay(ms);
}

void ArduinoImpl::SetPinMode(uint8_t pin, const PinMode& mode) const {
  pinMode(pin, uint8_t(mode));
}

} // namespace arduino
