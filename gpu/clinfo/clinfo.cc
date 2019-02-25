// Print a OpenClDevices proto describing available OpenCL devices.

#include <iostream>

#include "gpu/clinfo/proto/clinfo.pb.h"
#include "gpu/clinfo/libclinfo.h"


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
