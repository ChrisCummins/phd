// Get information about available OpenCL devices.

#ifndef PHD_GPU_LIBCLINFO_H
#define PHD_GPU_LIBCLINFO_H

#include <vector>

// Enable OpenCL cl::Error class and exceptions.
// This must be defined before including the OpenCL header.
#define __CL_ENABLE_EXCEPTIONS

#include "third_party/opencl/include/cl.hpp"

#include "gpu/clinfo/proto/clinfo.pb.h"

namespace phd {

namespace gpu {

namespace clinfo {

const char *OpenClErrorString(cl_int err);

void OpenClCheckError(const char *api_call, cl_int err);

void SetOpenClDevice(const cl::Platform &platform, const cl::Device &device,
                     ::gpu::clinfo::OpenClDevice *const message);

::gpu::clinfo::OpenClDevices GetOpenClDevices();

}  // namespace clinfo

}  // namespace gpu

}  // namespace phd

#endif //PHD_GPU_LIBCLINFO_H
