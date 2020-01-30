// Copyright (c) 2016-2020 Chris Cummins.
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
#include "gpu/cldrive/testutil.h"

namespace gpu {
namespace cldrive {
namespace test {

cl::Kernel CreateClKernel(const string& opencl_kernel) {
  cl::Program program(opencl_kernel);
  program.build("-cl-kernel-arg-info");

  std::vector<cl::Kernel> kernels;
  program.createKernels(&kernels);
  CHECK(kernels.size() == 1);
  return kernels[0];
}

DynamicParams MakeParams(size_t global_size, size_t local_size) {
  DynamicParams params;
  params.set_global_size_x(global_size);
  params.set_local_size_x(local_size);
  return params;
}

}  // namespace test
}  // namespace cldrive
}  // namespace gpu
