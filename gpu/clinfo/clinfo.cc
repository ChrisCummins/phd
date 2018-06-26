// Print a OpenClDevices proto describing available OpenCL devices.

#include <iostream>

#include "gpu/clinfo/libclinfo.h"


int main(int argc, char **argv) {
  try {
    std::cout << phd::gpu::clinfo::GetOpenClDevices().DebugString();
  } catch (cl::Error err) {
    std::cerr << "OpenCL Error: " << err.what() << " returned "
              << phd::gpu::clinfo::OpenClErrorString(err.err()) << std::endl;
    exit(1);
  }
}
