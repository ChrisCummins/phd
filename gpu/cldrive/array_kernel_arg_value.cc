#include "gpu/cldrive/array_kernel_arg_value.h"

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

// template <>
///*virtual*/ string ArrayKernelArgValue<phd::int32>::ToString() const
////*override*/ {
// return absl::StrFormat("%d", value_);
//}
//
// template <>
///*virtual*/ string ArrayKernelArgValue<phd::int64>::ToString() const
////*override*/ {
// return absl::StrFormat("%d", value_);
//}
//
// template <>
///*virtual*/ string ArrayKernelArgValue<float>::ToString() const /*override*/ {
// return absl::StrFormat("%.3f", value_);
//}
//
// template <>
///*virtual*/ string ArrayKernelArgValue<double>::ToString() const /*override*/
///{
// return absl::StrFormat("%.3f", value_);
//}

}  // namespace cldrive
}  // namespace gpu
