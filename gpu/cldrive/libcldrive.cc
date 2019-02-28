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
#include "gpu/cldrive/libcldrive.h"

#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/kernel_driver.h"
#include "gpu/clinfo/libclinfo.h"

#include "phd/logging.h"
#include "phd/status.h"
#include "phd/statusor.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#define LOG_CL_ERROR(level, error)                                  \
  LOG(level) << "OpenCL exception: " << error.what() << ", error: " \
             << phd::gpu::clinfo::OpenClErrorString(error.err());

namespace gpu {
namespace cldrive {

namespace {

// Attempt to build OpenCL program.
phd::StatusOr<cl::Program> BuildOpenClProgram(const std::string& opencl_kernel,
                                              const cl::Context& context,
                                              const string& cl_build_opts) {
  auto start_time = absl::Now();
  try {
    // Assemble the build options. We need -cl-kernel-arg-info so that we can
    // read the kernel signatures.
    string all_build_opts = "-cl-kernel-arg-info ";
    absl::StrAppend(&all_build_opts, cl_build_opts);
    phd::TrimRight(all_build_opts);

    cl::Program program(context, opencl_kernel);
    program.build(context.getInfo<CL_CONTEXT_DEVICES>(),
                  all_build_opts.c_str());
    auto end_time = absl::Now();
    auto duration = (end_time - start_time) / absl::Milliseconds(1);
    LOG(INFO) << "clBuildProgram() with options '" << all_build_opts
              << "' completed in " << duration << " ms";
    return program;
  } catch (cl::Error e) {
    LOG_CL_ERROR(WARNING, e);
    return phd::Status(phd::error::Code::INVALID_ARGUMENT,
                       "clBuildProgram failed");
  }
}

}  // namespace

Cldrive::Cldrive(CldriveInstance* instance, const cl::Device& device)
    : instance_(instance), device_(device) {}

void Cldrive::RunOrDie(const bool streaming_csv_output) {
  try {
    DoRunOrDie(streaming_csv_output);
  } catch (cl::Error error) {
    LOG(FATAL) << "Unhandled OpenCL exception.\n"
               << "    Raised by:  " << error.what() << '\n'
               << "    Error code: " << error.err() << " ("
               << phd::gpu::clinfo::OpenClErrorString(error.err()) << ")\n"
               << "This is a bug! Please report to "
               << "<https://github.com/ChrisCummins/cldrive/issues>.";
  }
}

void Cldrive::DoRunOrDie(const bool streaming_csv_output) {
  cl::Context context(device_);
  cl::CommandQueue queue(context,
                         /*devices=*/context.getInfo<CL_CONTEXT_DEVICES>()[0],
                         /*properties=*/CL_QUEUE_PROFILING_ENABLE);

  // Compile program or fail.
  phd::StatusOr<cl::Program> program_or = BuildOpenClProgram(
      string(instance_->opencl_src()), context, instance_->build_opts());
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
    KernelDriver(context, queue, kernel, instance_)
        .RunOrDie(streaming_csv_output);
  }

  instance_->set_outcome(CldriveInstance::PASS);
}

void ProcessCldriveInstancesOrDie(CldriveInstances* instance) {
  auto device = phd::gpu::clinfo::GetOpenClDeviceOrDie(instance->device());
  for (int i = 0; i < instance->instance_size(); ++i) {
    Cldrive(&instance->instance(i), device).RunOrDie(false);
  }
}

}  // namespace cldrive
}  // namespace gpu
