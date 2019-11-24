#include "maker/arduino/interface/wifi_server.h"

namespace arduino {

class WiFiServerClient {};

void ESP8266WiFiServer::Init(bool serial_print) {}

WiFiServerClient ESP8266WiFiServer::BlockUntilClientConnected() {
  return WiFiServerClient();
}

}  // namespace arduino
