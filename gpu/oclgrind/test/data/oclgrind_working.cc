// Check that only a single Oclgrind platform is available, with a single
// Oclgrind device.
#include <iostream>
#include <string>
#include <vector>

#include "phd/logging.h"
#include "third_party/opencl/cl.hpp"

int main(void) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (const auto& platform : platforms) {
    string str;

    platform.getInfo(CL_PLATFORM_NAME, &str);
    LOG(INFO) << "Platform name: " << str;

    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    CHECK(str.find("oclgrind") != string::npos)
        << "Expected platform name 'Oclgrind', found '" << str << "'";

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (const auto& device : devices) {
      device.getInfo(CL_DEVICE_NAME, &str);
      LOG(INFO) << "Device name: " << str;

      std::transform(str.begin(), str.end(), str.begin(), ::tolower);
      CHECK(str.find("oclgrind") != string::npos)
          << "Expected device name 'Oclgrind', found '" << str << "'";
    }
    CHECK(devices.size() == 1)
        << "Expected 1 device, found: " << devices.size();
  }

  CHECK(platforms.size() == 1)
      << "Expected 1 platform, found: " << platforms.size();

  LOG(INFO) << "done";

  return 0;
}
