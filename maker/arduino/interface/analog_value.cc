#ifdef ARDUINO
#include <Arduino_interface_analog_value.h>
#else
#include "maker/arduino/interface/analog_value.h"
#endif

namespace arduino {

/* static */ AnalogValue AnalogValue::Min() { return AnalogValue(0); }
/* static */ AnalogValue AnalogValue::Max() { return AnalogValue(255); }

}  // namespace arduino
