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
