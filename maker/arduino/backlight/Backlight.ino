// The control program for an LED strip, used as a backlight.
//
// TODO(cec): The Arduino is to be replaced by an ESP8266 and the lights
// controlled by WiFi. This requires a 74HCT245 logic level shifter.
//
// The circuit is:
//    * Arduino powered through USB.
//    * WS2812B LED strip +V/GND connected to beefy 5V DC power supply.
//    * 100uF capacitor between +V/GND of WS2812B LED strip.
//    * WS2812B LED strip GND connected to micro-controller GND.
//    * Micro-controller kDataPin connected to WS2812B LED strip data pin via
//      330 Ohm resistor.
//
#include <Adafruit_NeoPixel.h>

// The pin that the LED
const int kDataPin = 15;
// const int kDataPin = 13;
// The number of LEDs in the strip. This is
const int kLedCount = 83;

// The LED strip.
Adafruit_NeoPixel strip =
    Adafruit_NeoPixel(kLedCount, kDataPin, NEO_GRB + NEO_KHZ800);

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  strip.begin();
  strip.fill(strip.Color(255, 255, 255));
  strip.setBrightness(5);
  strip.show();
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
}
