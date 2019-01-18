// This sketch connects to a network over WiFi, create a server with a static
// IP, and exposes GET requests to control an LED:
//
//     http://<server>/led/on
//     http://<server>/led/off
//     http://<server>/led/brightness/<0-255>
//
// The server responds with a simple JSON response. Invalid URLs return 404.
// Example:
//
//     $ curl http://<server>/led/brightness/128
//     { "message": "LED on" }
//
// To save defining the SSID and password of the router to connect to in this
// file, I have #defined MY_ROUTER_SSID and MY_ROUTER_PASSWORD macros, and
// appended them to the flags of all ESP8266 builds. To do this, I modified the
// file: ~/Library/Arduino15/packages/esp8266/hardware/esp8266/2.4.2/platform.txt
// and appended the arguments to the C compiler and preprocessor flags:
//
//     compiler.cpreprocessor.flags=<snip> -DMY_ROUTER_SSID="foo" -DMY_ROUTER_PASSWORD="bar"
//     compiler.c.flags=<snip> -DMY_ROUTER_SSID="foo" -DMY_ROUTER_PASSWORD="bar"
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
// Homebridge and the homebridge-http package:
//
//     $ sudo apt-get install libavahi-compat-libdnssd-dev
//     $ npm install -g homebridge homebridge-http
//
// Then add an "accessories" entry to ~/.homebridge/config.json with the URLs
// handled by this sketch:
//
//     {
//       "bridge":{
//         "name":"Homebridge",
//         "username":"CC:22:3D:E3:CE:30",
//         "port":51826,
//         "pin":"031-45-154"
//       },
//       "platforms":[
//       ],
//       "accessories":[
//         {
//           "accessory":"Http",
//           "name":"Circuit Board",
//           "on_url":"http://192.168.0.225/led/on",
//           "off_url":"http://192.168.0.225/led/off",
//           "http_method":"GET"
//         }
//       ]
//     }
//
#include <Arduino.h>
#include <ESP8266WiFi.h>

#include "HttpJsonResponse.h"

// Static IP configuration.
const IPAddress ip(192, 168, 0, 225);
const IPAddress gateway(192, 168, 0, 1);
const IPAddress subnet(255, 255, 255, 0);

// The pin of the LED to control.
const int led_pin = 15;

// Create an instance of the server
// specify the port to listen on as an argument
WiFiServer server(80);

String WiFiStatusToString(int status) {
  switch (status) {
    case WL_CONNECT_FAILED:
      return "WL_CONNECT_FAILED";
    case WL_NO_SSID_AVAIL:
      return "WL_NO_SSID_AVAIL";
    case WL_CONNECTION_LOST:
      return "WL_CONNECTION_LOST";
    case WL_DISCONNECTED:
      return "WL_DISCONNECTED";
    case WL_IDLE_STATUS:
      return "WL_IDLE_STATUS";
    default:
      return String("unknown status (") + status + ")";
  }
}

void setup() {
  // Open serial port.
  Serial.begin(115200);
  delay(10);

  // Prepare LED for PWM output.
  analogWriteRange(255);
  pinMode(led_pin, OUTPUT);
  digitalWrite(led_pin, LOW);

  // Connect to WiFi network.
  WiFi.config(ip, gateway, subnet);
  WiFi.begin(MY_ROUTER_SSID, MY_ROUTER_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print("Connecting to ");
    Serial.print(MY_ROUTER_SSID);
    Serial.print(" ");
    Serial.println(WiFiStatusToString(WiFi.status()));
  }
  Serial.print("Connected to ");
  Serial.println(MY_ROUTER_SSID);

  // Start the server
  server.begin();

  Serial.print("Listening on ");
  Serial.println(WiFi.localIP());

  // I've read that this can help with issues where people were unable to
  // connect to the server after a couple of days. I'll leave it disabled for
  // now until I think I need it.
  // WiFi.setSleepMode(WIFI_NONE_SLEEP);
}

// Handle a HTTP request line and return a response.
HttpJsonResponse HandleRequest(const String& request_line) {
  HttpJsonResponse response_not_found = {
    "404 Not Found",
    "{ \"message\": \"Invalid request!\" }"
  };

  if (request_line.startsWith("GET /led/off ")) {
    digitalWrite(led_pin, LOW);
    return HttpJsonResponse {"200 OK", "{ \"message\": \"LED off\" }"};
  } else if (request_line.startsWith("GET /led/on ")) {
    digitalWrite(led_pin, HIGH);
    return HttpJsonResponse {"200 OK", "{ \"message\": \"LED on\" }"};
  } else if (request_line.startsWith("GET /led/brightness/")) {
    // Strip the start of the request.
    String tail = request_line.substring(sizeof("GET /led/brightness/") - 1);

    // Filter invalid request lines:
    if (!tail.length() ||  // Empty string
        !isDigit(tail.charAt(0)) ||  // Non-numeric digit.
        tail.indexOf(' ') == -1) {  // Request has no suffix.
      return response_not_found;
    }

    // Parse integer brightness.
    const int new_brightness = tail.toInt();
    if (0 <= new_brightness && new_brightness <= 255) {
      // If requested brightness is in range, set it.
      analogWrite(led_pin, new_brightness);
      return HttpJsonResponse{
          "200 OK",
          (String("{ \"message \": \"Setting LED brightness to ") +
           new_brightness + "\" }")
      };
    } else {
      // Else brightness is invalid.
      return response_not_found;
    }
  } else {
    return response_not_found;
  }
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (!client) {
    return;
  }

  // Wait until the client sends some data
  while(!client.available()) {
    delay(1);
  }

  // Read the first line of the request
  String req = client.readStringUntil('\r');

  // Debug print
  Serial.print("T[");
  Serial.print(millis());
  Serial.print("] request: `");
  Serial.print(req);
  Serial.print("` -> ");
  client.flush();

  // Process request and return response.
  auto response = HandleRequest(req);

  Serial.println(response.response_line);
  response.SendToClient(client);
}
