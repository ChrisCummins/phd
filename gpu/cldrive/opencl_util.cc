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
