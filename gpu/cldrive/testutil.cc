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

DynamicParams MakeParams(size_t global_size, size_t local_size = 1) {
  DynamicParams params;
  params.set_global_size_x(global_size);
  params.set_global_size_x(local_size);
  return params;
}

}  // namespace test
}  // namespace cldrive
}  // namespace gpu
