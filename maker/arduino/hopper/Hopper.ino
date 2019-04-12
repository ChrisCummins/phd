// This is the controller program for hopper, a set of 26 parallel LEDs. It
// creates a server with a static IP, then expose GET requests to control the
// LEDs using PWM:
//
//     http://<server>/hopper/lights/get/on_off
//     http://<server>/hopper/lights/set/on_off/{0,1}
//     http://<server>/hopper/lights/get/brightness
//     http://<server>/hopper/lights/set/brightness/[0,100]
//
// The server responds with pltainext responses. Invalid URLs return 404.
// For example, to set the LED to approximate half brightness:
//
//     $ curl http://<server>/hopper/lights/set/brightness/50
//
// To save defining the SSID and password of the router to connect to in this
// file, I have #defined MY_ROUTER_SSID and MY_ROUTER_PASSWORD macros, and
// appended them to the flags of all ESP8266 builds. To do this, I modified the
// file:
// ~/Library/Arduino15/packages/esp8266/hardware/esp8266/2.4.2/platform.txt and
// appended the arguments to the C compiler and preprocessor flags:
//
//     compiler.cpreprocessor.flags=<snip> -DMY_ROUTER_SSID="foo"
//     -DMY_ROUTER_PASSWORD="bar" compiler.c.flags=<snip> -DMY_ROUTER_SSID="foo"
//     -DMY_ROUTER_PASSWORD="bar"
//
// Configure Arduino IDE to build this sketch using:
//
//     Tools -> Board -> NodeMCU 1.0 (ESP-12E Module)
//     Tools -> Flash Size -> 4M (3M SPIFFS)
//     Tools -> CPU Frequency -> 80 Mhz
//     Tools -> Upload Speed -> 921600
//     Tools-->Port--> (whatever it is)
//
// To interface this sketch with HomeKit for "Hey Siri" support, install
// Homebridge and the homebridge-http package on an always-on server that will
// act as your "bridge":
//
//     $ sudo apt-get install libavahi-compat-libdnssd-dev
//     $ npm install -g homebridge homebridge-minimal-http-lightbulb-brightness
//
// Then add an "accessories" entry to ~/.homebridge/config.json with the URLs
// handled by this sketch:
//
//     {
//       "bridge": {
//         "name": "Demo",
//         "username": "AA:BB:CC:DD:EE:FF",
//         "port": 51826,
//         "pin": "123-45-678"
//       },
//       "platforms":[],
//       "accessories":[
//         {
//           "name":"Hopper",
//           "accessory":"MinimalisticHttpLightbulbBrightness",
//           "get_on_off_url": "http://192.168.0.225/hopper/lights/get/on_off/",
//           "set_on_off_url":
//           "http://192.168.0.225/hopper/lights/set/on_off/%on_off%",
//           "get_brightness_url":
//           "http://192.168.0.225/hopper/lights/get/brightness/",
//           "set_brightness_url":
//           "http://192.168.0.225/hopper/lights/set/brightness/%brightness%",
//           "get_on_off_polling_millis": "500",
//           "get_brightness_polling_millis": "500"
//         }
//       ]
//     }
//
#include <Arduino.h>
#include <ESP8266WiFi.h>

#include "HopperAction.h"
#include "HttpUtil.h"
#include "SerialUtil.h"

// Static IP configuration.
const IPAddress kIpAddress(192, 168, 0, 225);
const IPAddress kGateway(192, 168, 0, 1);
const IPAddress kSubnet(255, 255, 255, 0);

const IPAddress ip(192, 168, 0, 225);
const IPAddress gateway(192, 168, 0, 1);
const IPAddress subnet(255, 255, 255, 0);

const int kPort = 80;

const char* kSsid = MY_ROUTER_SSID;
const char* kPassword = MY_ROUTER_PASSWORD;

// The pin of the LED to control.
const int kLedPin = 15;

// Mutable state.

// Keep track of the current and "persistent" brightnesses. The
// current_brightness is what the brightnesses of the LEDs actually is. The
// "persistent" brightness is the last non-zero brightness of the LEDs. If
// current_brightness != 0, current_brightness == persistent_brightness.
int current_brightness = 0;
int persistent_brightness = 100;

// Configurable fading speed and time.
const int kBrightnessIncreaseStepSize = 1;
const int kBrightnessDecreaseStepSize = 2;
const int kBrightnessStepDuration_ms = 5;

// Create an instance of the server
// specify the port to listen on as an argument
WiFiServer server(kPort);

namespace {

// Smoothly transition between start and end value. Returns end value.
void SmoothPwmTransition(const uint8_t pin, const int start, const int end,
                         const int increase_step_size,
                         const int decrease_step_size,
                         const int step_duration_ms) {
  int current = start;
  while (current != end) {
    if (current < end) {
      current += increase_step_size;
      // Overshoot.
      if (current > 100) {
        current = 100;
      }
    } else if (current > end) {
      current -= decrease_step_size;
      // Undershoot.
      if (current < 0) {
        current = 0;
      }
    }
    analogWrite(pin, current);
    delay(step_duration_ms);
  }
}

}  // anonymous namespace

void setup() {
  // Open serial port.
  Serial.begin(115200);
  // Warning: Changing this delay to a larger value will cause Wifi.begin() to
  // get stuck in a perpetual state of WL_DISCONNECTED and not connect.
  delay(10);

  // Prepare LED for PWM output.
  analogWriteRange(100);
  pinMode(kLedPin, OUTPUT);
  analogWrite(kLedPin, current_brightness);

  // Connect to WiFi network.
  serial::Print("Connecting to ", kSsid, " ...\n");

  WiFi.config(kIpAddress, kGateway, kSubnet);
  WiFi.begin(kSsid, kPassword);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1500);
    serial::PrintLn(millis(), " ", http::WiFiStatusToString(WiFi.status()));
  }

  // Start the server
  server.begin();

  serial::PrintLn("Connected and listening on ", WiFi.localIP(), ":", kPort);

  // I've read that this can help with issues where people were unable to
  // connect to the server after a couple of days. I'll leave it disabled for
  // now until I think I need it.
  // WiFi.setSleepMode(WIFI_NONE_SLEEP);
}

// Handle a HTTP request line and return a response.
HopperAction HandleRequest(const String& request_line,
                           const int current_brightness,
                           const int persistent_brightness) {
  HopperAction response_not_found("404 Not Found", "text/plain",
                                  "Invalid request!", current_brightness);

  // GET /hopper/lights/get/on_off to get lights.
  if (request_line.startsWith("GET /hopper/lights/get/on_off")) {
    return HopperAction("200 OK", "text/plain", current_brightness ? "1" : "0",
                        current_brightness);
  }

  // POST /hopper/lights/set/on_off/{0,1} to set lights.
  else if (request_line.startsWith("POST /hopper/lights/set/on_off/")) {
    String tail =
        request_line.substring(sizeof("POST /hopper/lights/set/on_off/") - 1);
    if (tail.startsWith("true ")) {
      return HopperAction("204 No Content", "text/plain", "",
                          persistent_brightness);
    } else if (tail.startsWith("false ")) {
      return HopperAction("204 No Content", "text/plain", "", 0);
    }
  }

  // GET /hopper/lights/get/brightness to get brightness in range [0,100].
  else if (request_line.startsWith("GET /hopper/lights/get/brightness")) {
    return HopperAction("200 OK", "text/plain", String(persistent_brightness),
                        current_brightness);
  }

  // POST /hopper/lights/set/brightness/[0,100] to set brightness.
  else if (request_line.startsWith("POST /hopper/lights/set/brightness/")) {
    String tail = request_line.substring(
        sizeof("POST /hopper/lights/set/brightness/") - 1);
    const int new_brightness = tail.toInt();
    if (0 <= new_brightness && new_brightness <= 100) {
      return HopperAction("204 No Content", "text/plain", "", new_brightness);
    }
  }

  return response_not_found;
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (!client) {
    return;
  }

  // Wait until the client sends some data
  while (!client.available()) {
    delay(1);
  }

  // Read the first line of the request
  String req = client.readStringUntil('\r');

  // Debug print
  serial::Print("T[", millis(), "] request: `", req, "` -> ");

  // Process request and send the response to client.
  client.flush();
  HopperAction action =
      HandleRequest(req, current_brightness, persistent_brightness);
  client.print(action.GetHttpResponse().ToString());
  serial::PrintLn(action.GetHttpResponse().GetResponseLine());

  SmoothPwmTransition(kLedPin, current_brightness, action.GetNewBrightness(),
                      kBrightnessIncreaseStepSize, kBrightnessDecreaseStepSize,
                      kBrightnessStepDuration_ms);
  current_brightness = action.GetNewBrightness();
  if (current_brightness) {
    persistent_brightness = current_brightness;
  }
}
