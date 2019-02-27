#include "gpu/cldrive/opencl_util.h"

namespace gpu {
namespace cldrive {
namespace util {

string GetOpenClKernelName(const cl::Kernel& kernel) {
  string name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
  DCHECK(name.size() >= 2);
  return name.substr(0, name.size() - 1);
}

}  // namespace util
}  // namespace cldrive
}  // namespace gpu
