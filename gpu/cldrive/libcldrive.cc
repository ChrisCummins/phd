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

#include "absl/time/time.h"
#include "absl/time/clock.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#define LOG_CL_ERROR(level, error)                                  \
  LOG(level) << "OpenCL exception: " << error.what() << ", error: " \
             << phd::gpu::clinfo::OpenClErrorString(error.err());

namespace gpu {
namespace cldrive {

namespace {

// Attempt to build OpenCL program.
phd::StatusOr<cl::Program> BuildOpenClProgram(
    const std::string& opencl_kernel, const std::vector<cl::Device>& devices) {
  auto start_time = absl::Now();
  try {
    // Assemble the build options. We ned -cl-kernel-arg-info so that we can
    // read the kernel signatures.
    string build_opts = "-cl-kernel-arg-info "
    absl::StrAppend(&build_opts, build_opts.c_str());

    cl::Program program(opencl_kernel);
    program.build(devices, build_opts);
    auto end_time = absl::Now();
    auto duration = (end_time - start_time) / absl::Milliseconds(1);
    LOG(INFO) << "OpenCL program build completed in " << duration << " ms";
    return program;
  } catch (cl::Error e) {
    LOG_CL_ERROR(ERROR, e);
    return phd::Status::UNKNOWN;
  }
}

}  // namespace

Cldrive::Cldrive(CldriveInstance* instance, const cl::Device& device)
      : instance_(instance),
        device_(device),
        context_(device_),
        queue_(context_, /*devices=*/context_.getInfo<CL_CONTEXT_DEVICES>()[0],
               /*properties=*/CL_QUEUE_PROFILING_ENABLE) {}

void Cldrive::RunOrDie(const bool streaming_csv_output) {
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
    KernelDriver(context_, queue_, kernel, instance_).RunOrDie(streaming_csv_output);
  }

  instance_->set_outcome(CldriveInstance::PASS);
}


void ProcessCldriveInstanceOrDie(CldriveInstance* instance) {
  try {
    auto device = phd::gpu::clinfo::GetOpenClDeviceOrDie(instance->device());
    Cldrive(instance, device).RunOrDie();
  } catch (cl::Error error) {
    LOG_CL_ERROR(FATAL, error);
  }
}

}  // namespace cldrive
}  // namespace gpu
