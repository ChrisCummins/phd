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
#pragma once

#include "gpu/cldrive/kernel_arg_set.h"
#include "gpu/cldrive/logger.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/statusor.h"
#include "phd/string.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelDriver {
 public:
  KernelDriver(const cl::Context& context, const cl::CommandQueue& queue,
               const cl::Kernel& kernel, CldriveInstance* instance,
               int instance_num);

  void RunOrDie(Logger& logger);

  phd::StatusOr<CldriveKernelRun> RunDynamicParams(
      const DynamicParams& dynamic_params, Logger& logger);

  gpu::libcecl::OpenClKernelInvocation RunOnceOrDie(
      const DynamicParams& dynamic_params, KernelArgValuesSet& inputs,
      KernelArgValuesSet* outputs, const CldriveKernelRun* const run,
      Logger& logger);

 private:
  // Private helper to public RunDynamicParams() method that doesn't catch
  // OpenCL exceptions.
  phd::Status RunDynamicParams(const DynamicParams& dynamic_params,
                               Logger& logger, CldriveKernelRun* run);

  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Device device_;
  cl::Kernel kernel_;
  const CldriveInstance& instance_;
  int instance_num_;
  CldriveKernelInstance* kernel_instance_;
  string name_;
  KernelArgSet args_set_;
};

}  // namespace cldrive
}  // namespace gpu
