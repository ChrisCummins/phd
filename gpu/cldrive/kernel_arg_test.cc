#include "gpu/cldrive/kernel_arg.h"

#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/proto/cldrive.pb.h"

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/test.h"

namespace gpu {
namespace cldrive {
namespace {

cl::Kernel KernelFromString(const string& opencl_kernel) {
  cl::Program program(opencl_kernel);
  program.build("-cl-kernel-arg-info");

  std::vector<cl::Kernel> kernels;
  program.createKernels(&kernels);
  CHECK(kernels.size() == 1);
  return kernels[0];
}

template<typename T>
ArrayKernelArgValue<T>* DowncastOrDie(KernelArgValue* t) {
  return dynamic_cast<ArrayKernelArgValue<T>*>(t);
}

TEST(KernelArg, GlobalPointerIsGlobal) {
  cl::Kernel kernel = KernelFromString("kernel void A(global int* a) {}");

  KernelArg arg(&kernel, 0);
  ASSERT_TRUE(arg.Init().ok());
  EXPECT_TRUE(arg.IsGlobal());
}

TEST(KernelArg, TryToCreateOnesValueGlobalInt) {
  // TODO(cec): Document, tidy up and expand.
  cl::Kernel kernel = KernelFromString("kernel void A(global int* a) {}");

  cl::Context context = cl::Context::getDefault();

  DynamicParams dynamic_params;
  dynamic_params.set_global_size_x(50);

  KernelArg arg(&kernel, 0);
  ASSERT_TRUE(arg.Init().ok());

  auto value = arg.TryToCreateOnesValue(context, dynamic_params);
  ASSERT_TRUE(value);

  auto array_value = DowncastOrDie<phd::int32>(value.get());
  ASSERT_TRUE(array_value);
  EXPECT_EQ(array_value->size(), 50);
}

}  // anonymous namespace
}  // namespace cldrive
}  // namespace gpu

TEST_MAIN();
