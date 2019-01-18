#include <LcdHelloWorld.h>

#include <Arduino_hardware.h>
#include <LiquidCrystal.h>
#include <Arduino.h>

arduino::LcdHelloWorld<arduino::ArduinoImpl> program;

void setup() {
  program.Setup();
}

void loop() {
  program.Loop();
}
