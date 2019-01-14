#include "maker/arduino/interface/pin_mode.h"

namespace arduino {

/* static */ PinMode PinMode::Output() { return PinMode(0); }
/* static */ PinMode PinMode::Input() { return PinMode(1); }
/* static */ PinMode PinMode::InputPullup() { return PinMode(2); }

}  // namespace arduino
