// An Arduino program which flashes the S-O-S sign in Morse Code.
#pragma once

// If compiling for hardware, use the generated library header. For C++
// compilation, use the bazel header path.
#ifdef ARDUINO
#include <Arduino_interface.h>
#include <LiquidCrystal.h>
#else
#include "maker/arduino/interface/arduino_interface.h"
#include "maker/arduino/interface/mock_liquid_crystal.h"
#endif

namespace arduino {

// The circuit:
// * LCD RS pin to digital pin 7
// * LCD Enable pin to digital pin 8
// * LCD D4 pin to digital pin 9
// * LCD D5 pin to digital pin 10
// * LCD D6 pin to digital pin 11
// * LCD D7 pin to digital pin 12
// * LCD R/W pin to ground
// * LCD VSS pin to ground
// * LCD VCC pin to 5V
// * 10K potentiometer:
//    * ends to +5V and ground
//    * wiper to LCD VO pin (pin 3)

// Pin connections.
const int rs = 7, en = 8, d4 = 9, d5 = 10, d6 = 11, d7 = 12;

// The program to execute.
template <typename ArduinoInterface>
class LcdHelloWorld {
 public:
  explicit LcdHelloWorld() : interface_(), lcd_(rs, en, d4, d5, d6, d7) {}

  void Setup() {
    // Initialize the LCD.
    lcd_.begin(16, 2);
    lcd_.print("Elapsed time:");
  }

  void Loop() {
    // Cursor position is zero indexed by column, row:
    lcd_.setCursor(0, 1);
    lcd_.print(interface().Millis() % 1000);
  }

  const ArduinoInterface& interface() const { return interface_; }
  const LiquidCrystal& lcd() const { return lcd_; }

 private:
  const ArduinoInterface interface_;
  LiquidCrystal lcd_;
};

}  // namespace arduino
