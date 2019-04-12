#include "maker/arduino/interface/wifi_server.h"

namespace arduino {

class WiFiServerClient {};

void WiFiServer::Init(bool serial_print) {}

WiFiServerClient WiFiServer::BlockUntilClientConnected() const {
  return WiFiServerClient();
}

}  // namespace arduino
