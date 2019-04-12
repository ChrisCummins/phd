#pragma once

#include <Arduino.h>

namespace http {

// Simple container for HTTP responses. I have to define it in a separate
// header file because Arduino toolchain doesn't let you return custom classes
// from functions, unless they are defined in libraries.
class HttpResponse {
 public:
  HttpResponse(const String& response_line, const String& content_type,
               const String& text_string)
      : response_line_(response_line),
        content_type_(content_type),
        text_string_(text_string) {}

  String ToString() const {
    return (String("HTTP/1.1 ") + response_line_ +
            "\r\n"
            "Server: Hopper\r\n"
            "Content-Type: " +
            content_type_ + "\r\n\r\n" + text_string_ + "\r\n");
  }

  const String& GetResponseLine() const { return response_line_; }

 private:
  const String response_line_;
  const String content_type_;
  const String text_string_;
};

const char* WiFiStatusToString(int status) {
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
      return "Unknown status";
  }
}

}  // namespace http
