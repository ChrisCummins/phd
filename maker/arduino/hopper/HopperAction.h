#pragma once

#include <Arduino.h>
#include "HttpUtil.h"

class HopperAction {
 public:
  HopperAction(const String& response_line, const String& content_type,
               const String& text_string, const int new_brightness)
      : http_response_(response_line, content_type, text_string),
        new_brightness_(new_brightness) {}

  int GetNewBrightness() const { return new_brightness_; }

  const http::HttpResponse& GetHttpResponse() const { return http_response_; }

 private:
  const http::HttpResponse http_response_;
  const int new_brightness_;
};
