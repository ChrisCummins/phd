// Check that only a single Oclgrind platform is available, with a single
// Oclgrind device.
//
// Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
