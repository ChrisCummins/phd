#include "gpu/cldrive/kernel_arg.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/test.h"

namespace gpu {
namespace cldrive {
namespace {

cl::Kernel KernelFromString(const string& opencl_kernel) {
  cl::Program program(opencl_kernel);
  program.build("-cl-kernel-arg-info");

  std::vector<cl::Kernel> kernels;
  program.createKernels(&kernels);
  return kernels[0];
}

TEST(KernelArg, GlobalPointerIsGlobal) {
  cl::Kernel kernel = KernelFromString("kernel void A(global int* a) {}");

  KernelArg arg(&kernel, 0);
  ASSERT_TRUE(arg.Init().ok());
  EXPECT_TRUE(arg.IsGlobal());
}

}  // anonymous namespace
}  // namespace cldrive
}  // namespace gpu

TEST_MAIN();
