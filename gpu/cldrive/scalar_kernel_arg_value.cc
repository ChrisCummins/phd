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
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

template <>
/*virtual*/ string ScalarKernelArgValue<phd::int32>::ToString() const
/*override*/ {
  string s = "";
  absl::StrAppend(&s, value());
  return s;
}

template <>
/*virtual*/ string ScalarKernelArgValue<phd::int64>::ToString() const
/*override*/ {
  string s = "";
  absl::StrAppend(&s, value());
  return s;
}

template <>
/*virtual*/ string ScalarKernelArgValue<float>::ToString() const /*override*/ {
  string s = "";
  absl::StrAppend(&s, value());
  return s;
}

template <>
/*virtual*/ string ScalarKernelArgValue<double>::ToString() const /*override*/ {
  string s = "";
  absl::StrAppend(&s, value());
  return s;
}

}  // namespace cldrive
}  // namespace gpu
