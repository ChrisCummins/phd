// Get information about available OpenCL devices.

#ifndef PHD_GPU_LIBCLINFO_H
#define PHD_GPU_LIBCLINFO_H

#include <vector>

#include "third_party/opencl/cl.hpp"

#include "gpu/clinfo/proto/clinfo.pb.h"
#include "phd/statusor.h"
#include "phd/string.h"

// TODO(cec): Refactor to remove phd:: namespace.
namespace phd {

namespace gpu {

namespace clinfo {

// TODO(cec): Refactor so that each returns StatusOr, not exception.

const char* OpenClErrorString(cl_int err);

void OpenClCheckError(const string& api_call, cl_int err);

void SetOpenClDevice(const cl::Platform& platform, const cl::Device& device,
                     const int platform_id, const int device_id,
                     ::gpu::clinfo::OpenClDevice* const message);

::gpu::clinfo::OpenClDevices GetOpenClDevices();

::gpu::clinfo::OpenClDevice GetOpenClDevice(const int platform_id,
                                            const int device_id);

// Find and return the OpenCL device proto by name.
StatusOr<::gpu::clinfo::OpenClDevice> GetOpenClDeviceProto(const string& name);

// Lookup an OpenCL device by its proto representation. Raises
// std::invalid_argument if not found.
cl::Device GetOpenClDevice(const ::gpu::clinfo::OpenClDevice& device);

// Same as above, but abort if device is not available.
cl::Device GetOpenClDeviceOrDie(const ::gpu::clinfo::OpenClDevice& device);

// Lookup an OpenCL device by its name. Raises std::invalid_argument if not
// found.
cl::Device GetOpenClDevice(const string& name);

// Same as above, but abort if device is not available.
cl::Device GetOpenClDeviceOrDie(const string& name);

}  // namespace clinfo

}  // namespace gpu

}  // namespace phd

#endif  // PHD_GPU_LIBCLINFO_H
