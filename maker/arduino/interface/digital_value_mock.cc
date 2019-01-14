#include "maker/arduino/interface/digital_value.h"

namespace arduino {

/* static */ DigitalValue DigitalValue::High() { return DigitalValue(1); }
/* static */ DigitalValue DigitalValue::Low() { return DigitalValue(0); }

}  // namespace arduino
