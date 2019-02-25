// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#define __CL_ENABLE_EXCEPTIONS

#include "gpu/cldrive/libcldrive.h"

#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/kernel_driver.h"
#include "gpu/clinfo/libclinfo.h"

#include "phd/logging.h"
#include "phd/status.h"
#include "phd/statusor.h"

#include "absl/strings/str_format.h"
#include "third_party/opencl/include/cl.hpp"

#define LOG_CL_ERROR(level, error)                                  \
  LOG(level) << "OpenCL exception: " << error.what() << ", error: " \
             << phd::gpu::clinfo::OpenClErrorString(error.err());

namespace gpu {
namespace cldrive {

namespace {

// Attempt to build OpenCL program.
phd::StatusOr<cl::Program> BuildOpenClProgram(
    const std::string& opencl_kernel, const std::vector<cl::Device>& devices) {
  try {
    cl::Program program(opencl_kernel);
    program.build(devices, "-cl-kernel-arg-info");
    return program;
  } catch (cl::Error e) {
    LOG_CL_ERROR(ERROR, e);
    return phd::Status::UNKNOWN;
  }
}

}  // namespace

class Cldrive {
 public:
  explicit Cldrive(CldriveInstance* instance)
      : instance_(instance),
        device_(phd::gpu::clinfo::GetOpenClDeviceOrDie(instance->device())),
        context_(device_),
        queue_(context_, /*devices=*/context_.getInfo<CL_CONTEXT_DEVICES>()[0],
               /*properties=*/CL_QUEUE_PROFILING_ENABLE) {}

  void RunOrDie() {
    // Compile program or fail.
    phd::StatusOr<cl::Program> program_or =
        BuildOpenClProgram(string(instance_->opencl_src()),
                           context_.getInfo<CL_CONTEXT_DEVICES>());
    if (!program_or.ok()) {
      LOG(ERROR) << "OpenCL program compilation failed!";
      instance_->set_outcome(CldriveInstance::PROGRAM_COMPILATION_FAILURE);
      return;
    }
    cl::Program program = program_or.ValueOrDie();

    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);

    if (!kernels.size()) {
      LOG(ERROR) << "OpenCL program contains no kernels!";
      instance_->set_outcome(CldriveInstance::NO_KERNELS_IN_PROGRAM);
      return;
    }

    for (auto& kernel : kernels) {
      KernelDriver(context_, queue_, kernel, instance_).RunOrDie();
    }

    instance_->set_outcome(CldriveInstance::PASS);
  };

 private:
  CldriveInstance* instance_;
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
};

void ProcessCldriveInstanceOrDie(CldriveInstance* instance) {
  try {
    Cldrive(instance).RunOrDie();
  } catch (cl::Error error) {
    LOG_CL_ERROR(FATAL, error);
  }
}

}  // namespace cldrive
}  // namespace gpu
