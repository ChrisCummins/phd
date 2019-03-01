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
#include "gpu/cldrive/opencl_util.h"

namespace gpu {
namespace cldrive {
namespace util {

namespace {

// Workaround for a defect in which getInfo<>() methods return strings
// including terminating '\0' character. See discussion at:
// https://github.com/KhronosGroup/OpenCL-CLHPP/issues/8
void StripTrailingNullCharacter(string* str) {
  if (!str->empty() && str->back() == '\0') {
    str->resize(str->size() - 1);
  }
}

}  // anonymous namespace

string GetOpenClKernelName(const cl::Kernel& kernel) {
  string name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
  StripTrailingNullCharacter(&name);
  CHECK(name.size()) << "Empty string returned by getInfo()";
  return name;
}

string GetKernelArgTypeName(const cl::Kernel& kernel, size_t arg_index) {
  string name = kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
  StripTrailingNullCharacter(&name);
  CHECK(name.size()) << "Empty string returned by getArgInfo()";
  return name;
}

}  // namespace util
}  // namespace cldrive
}  // namespace gpu
