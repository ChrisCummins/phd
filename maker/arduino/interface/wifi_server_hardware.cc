#include <Arduino_interface_wifi_server.h>

#include <Arduino_interface.h>
#include <Arduino_interface_ip_address.h>

#include <Arduino.h>
#include <ESP8266WiFi.h>

#include <stdint.h>

namespace arduino {

namespace {

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

IPAddress MakeIpAddress(const IPv4Address& address) {
  return IPAddress(address.a(), address.b(), address.c(), address.d());
}

}  // anonymous namespace

class WiFiServerClient {
 public:
  explicit WiFiServerClient(const WiFiClient& client) : client_(client) {}

  void Print(const char* message) { client_.print(message); }

 private:
  WiFiClient client_;
};

void ESP8266WiFiServer::Init(bool serial_print) {
  WiFi.config(MakeIpAddress(ip_), MakeIpAddress(gateway_),
              MakeIpAddress(subnet_));
  WiFi.begin(ssid_, password_);
  while (WiFi.status() != WL_CONNECTED) {
    interface().Delay(1000);
    //    if (serial_print) {
    //      interface().SerialPrint("Connecting to ");
    //      interface().SerialPrint(ssid_);
    //      interface().SerialPrint(" ");
    //      interface().SerialPrint(WiFiStatusToString(WiFi.status()));
    //      interface().SerialPrint('\n');
    //    }
  }
  //  if (serial_print) {
  //    interface().SerialPrint("Connected to ");
  //    interface().SerialPrint(ssid_);
  //    interface().SerialPrint('\n');
  //  }

  // Start the server
  impl_.Begin();

  //  if (serial_print) {
  //    interface().SerialPrint("Listening on ");
  //    interface().SerialPrint(WiFi.localIP());
  //    interface().SerialPrint('\n');
  //  }

  // I've read that this can help with issues where people were unable to
  // connect to the server after a couple of days. I'll leave it disabled for
  // now until I think I need it.
  // WiFi.setSleepMode(WIFI_NONE_SLEEP);
}

WiFiServerClient ESP8266WiFiServer::BlockUntilClientConnected() {
  // Check if a client has connected
  WiFiClient client = impl_.Available();
  if (!client) {
    // TODO(cec): FIX ME!
    // return;
  }

  // Wait until the client sends some data
  while (!client.available()) {
    // interface().Delay(1);
  }
  return WiFiServerClient(client);
}

void HttpResponse::SendToClient(WiFiServerClient& client) {
  // Create a string response, format it, and return it to the client.
  client.Print("HTTP/1.1 ");
  client.Print(response_line_);
  client.Print("\r\n");
  client.Print("Server: Hopper\r\nContent-Type: ");
  client.Print(content_type_);
  client.Print("\r\n\r\n");
  client.Print(message_);
  client.Print("\r\n");
}

}  // namespace arduino
