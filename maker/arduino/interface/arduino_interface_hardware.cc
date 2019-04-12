#include <Arduino_hardware.h>

#include <Arduino.h>
#include <Arduino_interface.h>

namespace arduino {

ArduinoImpl::~ArduinoImpl() {}

DigitalValue ArduinoImpl::DigitalRead(const Pin& pin) const {
  return digitalRead(uint8_t(pin)) ? DigitalValue::High() : DigitalValue::Low();
}

void ArduinoImpl::DigitalWrite(const Pin& pin,
                               const DigitalValue& value) const {
  return digitalWrite(uint8_t(pin), uint8_t(value));
}

unsigned long ArduinoImpl::Millis(void) const { return millis(); }

void ArduinoImpl::Delay(unsigned long ms) const { delay(ms); }

void ArduinoImpl::SetPinMode(const Pin& pin, const PinMode& mode) const {
  pinMode(uint8_t(pin), uint8_t(mode));
}

void ArduinoImpl::InitSerial(unsigned long baud) const {
  Serial.begin(baud);
  delay(10);  // I've seen this "in the wild".
}

void ArduinoImpl::SerialPrint(const char* value) const { Serial.print(value); }

void ArduinoImpl::SetPwmRange(const int range) const {
  // analogWriteRange(range);
}

void ArduinoImpl::AnalogWrite(const Pin& pin, const AnalogValue& value) const {
  analogWrite(uint8_t(pin), uint8_t(value));
}

/* static */ const Pin ArduinoInterface::kBuiltInLedPin = Pin(LED_BUILTIN);

}  // namespace arduino
