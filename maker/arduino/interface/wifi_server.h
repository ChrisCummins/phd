#pragma once

#ifdef ARDUINO
#include <Arduino_interface.h>
#include <Arduino_interface_ip_address.h>
#else
#include "maker/arduino/interface/arduino_interface.h"
#include "maker/arduino/interface/ip_address.h"
#endif

#include <stdint.h>

namespace arduino {

class WiFiServerClient;

#ifdef ARDUINO
#include <Arduino.h>
#include <ESP8266WiFi.h>

class WiFiServerImpl {
 public:
  WiFiServerImpl(const ArduinoInterface& interface, const IPv4Address& ip,
                 const IPv4Address& gateway, const IPv4Address& subnet,
                 const char* ssid, const char* password, const int port)
      : server_(port) {}

  void Begin() { server_.begin(); }

  WiFiClient Available() { return server_.available(); }

 private:
  WiFiServer server_;
};
#else
class WiFiServerImpl {
 public:
  WiFiServerImpl(const ArduinoInterface& interface, const IPv4Address& ip,
                 const IPv4Address& gateway, const IPv4Address& subnet,
                 const char* ssid, const char* password, const int port) {}
};
#endif

class ESP8266WiFiServer {
 public:
  ESP8266WiFiServer(const ArduinoInterface& interface, const IPv4Address& ip,
                    const IPv4Address& gateway, const IPv4Address& subnet,
                    const char* ssid, const char* password, const int port)
      : interface_(interface),
        ip_(ip),
        gateway_(gateway),
        subnet_(subnet),
        ssid_(ssid),
        password_(password),
        port_(port),
        impl_(interface, ip, gateway, subnet, ssid, password, port) {}

  void Init(bool serial_print);

  WiFiServerClient BlockUntilClientConnected();

 private:
  const ArduinoInterface& interface() const { return interface_; }

  const ArduinoInterface& interface_;
  const IPv4Address& ip_;
  const IPv4Address& gateway_;
  const IPv4Address& subnet_;
  const char* ssid_;
  const char* password_;
  const int port_;

  WiFiServerImpl impl_;
};

// Simple container for HTTP responses.
class HttpResponse {
 public:
  HttpResponse(const char* response_line, const char* content_type,
               const char* message)
      : response_line_(response_line),
        content_type_(content_type),
        message_(message) {}

  void SendToClient(WiFiServerClient& client);

 private:
  const char* response_line_;
  const char* content_type_;
  const char* message_;
};

}  // namespace arduino
