#pragma once

#include <ESP8266WiFi.h>

// Simple container for HTTP responses. I have to define it in a separate
// header file because Arduino toolchain doesn't let you return custom classes
// from functions, unless they are defined in libraries.
class HttpJsonResponse {
 public:
  String response_line;
  String json_string;

  void SendToClient(WiFiClient& client) {
    // Create a string response, format it, and return it to the client.
    client.print(String("HTTP/1.1 ") + response_line +
                 "\r\n"
                 "Server: ESP12E\r\n"
                 "Content-Type: application/json\r\n\r\n" +
                 json_string + "\r\n");
  }
};
