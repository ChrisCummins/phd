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

#include "gpu/cldrive/kernel_arg.h"
#include "gpu/cldrive/kernel_arg_values_set.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/status.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArgSet {
 public:
  KernelArgSet(cl::Kernel* kernel);

  CldriveKernelInstance::KernelInstanceOutcome LogErrorOutcome(
      const CldriveKernelInstance::KernelInstanceOutcome& outcome);

  CldriveKernelInstance::KernelInstanceOutcome Init();

  phd::Status SetRandom(const cl::Context& context,
                        const DynamicParams& dynamic_params,
                        KernelArgValuesSet* values);

  phd::Status SetOnes(const cl::Context& context,
                      const DynamicParams& dynamic_params,
                      KernelArgValuesSet* values);

 private:
  cl::Kernel* kernel_;
  std::vector<KernelArg> args_;
};

}  // namespace cldrive
}  // namespace gpu
