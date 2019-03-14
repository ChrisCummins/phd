// Print a OpenClDevices proto describing available OpenCL devices.
//
// Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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

#include "gpu/clinfo/libclinfo.h"
#include "gpu/clinfo/proto/clinfo.pb.h"

int PrintHelp(char **argv) {
  std::cerr << "Usage: " << argv[0] << " [-p <platform_id> -d <device_id>]\n";
  return 1;
}

int main(int argc, char **argv) {
  int platform_id = -1, device_id = -1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      return PrintHelp(argv);
    else if (!strcmp(argv[i], "-p"))
      platform_id = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-d"))
      device_id = atoi(argv[++i]);
    else {
      fprintf(stderr, "Unrecognized argument '%s'\\n", argv[i]);
      return 1;
    }
  }

  try {
    if (platform_id != -1 && device_id != -1) {
      ::gpu::clinfo::OpenClDevices message;
      auto device = phd::gpu::clinfo::GetOpenClDevice(platform_id, device_id);
      auto new_device = message.add_device();
      new_device->CopyFrom(device);
      std::cout << message.DebugString();
    } else {
      std::cout << phd::gpu::clinfo::GetOpenClDevices().DebugString();
    }
  } catch (cl::Error err) {
    std::cerr << "OpenCL Error: " << err.what() << " returned "
              << phd::gpu::clinfo::OpenClErrorString(err.err()) << std::endl;
    return 1;
  }
}
