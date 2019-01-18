// Fading an LED using the analogWrite() function.

// The Circuit:
// * Led connected to board pin D8 (chip pin 15).

#include <Arduino.h>

// The PWM pin the LED is attached to.
const int led = 15;
// The range of PWM. For Arduino boards, PWM range is [0,255]. For ESP8266, PWM
// range is [0,1024].
const int pwm_range = 1024;

// The amount of time (in milliseconds) to delay for between changing
// brightness.
const int fade_delay = 30;

// How much to change the brightness by.
int fade_amount = (pwm_range / 100) * 2;

// Global state to track how bright the LED is.
int brightness = 0;

void setup() {
  pinMode(led, OUTPUT);
  // analogWriteRange() doesn't exist on Arduino boards.
  // analogWriteRange(pwm_range);
}

void loop() {
  // Set output.
  analogWrite(led, brightness);
  delay(fade_delay);

  // Update brightness.
  brightness += fade_amount;
  if (brightness <= 0 || brightness >= pwm_range) {
    // Reverse the direction of the fading at the ends of the fade.
    fade_amount = -fade_amount;
  }
}
