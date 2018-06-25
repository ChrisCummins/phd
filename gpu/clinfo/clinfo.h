//
// Created by Chris Cummins on 25/06/2018.
//

#ifndef PHD_GPU_CLINFO_H
#define PHD_GPU_CLINFO_H

#include <vector>

#include "gpu/clinfo/proto/clinfo.pb.h"

namespace phd {

namespace gpu {

namespace clinfo {


void OpenClCheckError(const char *api_call, cl_int err);

void OpenClCheckError(const char *api_call, cl_int err);

void SetOpenClDevice(const cl::Platform &platform, const cl::Device &device,
                     ::gpu::clinfo::OpenClDevice *const message);

::gpu::clinfo::OpenClDevices GetOpenClDevices();

}  // namespace clinfo

}  // namespace gpu

}  // namespace phd

#endif //PHD_GPU_CLINFO_H
