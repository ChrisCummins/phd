#include <Arduino_interface_digital_value.h>
#include <Arduino.h>

namespace arduino {

/* static */ DigitalValue DigitalValue::High() { return DigitalValue(HIGH); }
/* static */ DigitalValue DigitalValue::Low() { return DigitalValue(LOW); }

}  // namespace arduino
