// Get information about available OpenCL devices.
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

#ifndef PHD_GPU_LIBCLINFO_H
#define PHD_GPU_LIBCLINFO_H

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

// Return a "default" OpenCL device, where "default" will attempt to pick some
// safe device. If available, Oclgrind will be used. Else, the first device
// as it appears in the enumerated list.
// TODO(cec): It would be nice if we could link against oclgrind to ensure that
// this function could always return the oclgrind device.
cl::Device GetDefaultOpenClDeviceOrDie();

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
