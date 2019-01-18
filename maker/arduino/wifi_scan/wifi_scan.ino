// Scan for WiFi networks and return SSIDs over Serial.

#include <Arduino.h>
#include <ESP8266WiFi.h>

const int delay_between_scans_ms = 10000;

void setup() {
  Serial.begin(115200);

  // Set WiFi to station mode and disconnect from an AP if it was previously
  // connected.
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
}

void loop() {
  const int num_networks = WiFi.scanNetworks();

  Serial.print(num_networks);
  Serial.println(" networks found: ");

  for (int i = 0; i < num_networks; ++i) {
    // Print SSID and RSSI for each network found.
    Serial.print("  ");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(WiFi.SSID(i));
    Serial.print(" (");
    Serial.print(WiFi.RSSI(i));
    Serial.print(")");
    Serial.println((WiFi.encryptionType(i) == ENC_TYPE_NONE) ? "" : "*");
    delay(10);
  }

  Serial.println("");

  // Wait a bit before scanning again.
  delay(delay_between_scans_ms);
}
