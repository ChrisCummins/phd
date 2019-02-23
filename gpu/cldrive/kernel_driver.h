#pragma once

#include "gpu/cldrive/kernel_arg_set.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/statusor.h"
#include "phd/string.h"
#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelDriver {
 public:
  KernelDriver(const cl::Context& context, const cl::CommandQueue& queue,
               const cl::Kernel& kernel, CldriveInstance* instance);

  void RunOrDie();

  phd::StatusOr<CldriveKernelRun> CreateRunForParamsOrDie(
      const DynamicParams& dynamic_params, const bool output_checks);

  gpu::libcecl::OpenClKernelInvocation RunOnceOrDie(
      const DynamicParams& dynamic_params, const KernelArgValuesSet& inputs,
      KernelArgValuesSet* outputs);

 private:
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Device device_;
  cl::Kernel kernel_;
  CldriveInstance* instance_;
  CldriveKernelInstance* kernel_instance_;
  string name_;
  KernelArgSet args_set_;
};

}  // namespace cldrive
}  // namespace gpu
