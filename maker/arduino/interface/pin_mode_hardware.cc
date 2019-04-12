#include <Arduino.h>
#include <Arduino_interface_pin_mode.h>

namespace arduino {

/* static */ PinMode PinMode::Output() { return PinMode(OUTPUT); }
/* static */ PinMode PinMode::Input() { return PinMode(INPUT); }
/* static */ PinMode PinMode::InputPullup() { return PinMode(INPUT_PULLUP); }

}  // namespace arduino
