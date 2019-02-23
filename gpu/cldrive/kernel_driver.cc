#include "gpu/cldrive/kernel_driver.h"

namespace gpu {
namespace cldrive {

KernelDriver::KernelDriver(const cl::Context& context,
                           const cl::CommandQueue& queue,
                           const cl::Kernel& kernel, CldriveInstance* instance)
    : context_(context),
      queue_(queue),
      device_(context.getInfo<CL_CONTEXT_DEVICES>()[0]),
      kernel_(kernel),
      instance_(instance),
      kernel_instance_(instance->add_kernel()),
      name_(kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()),
      args_set_(context, &kernel_) {}

void KernelDriver::RunOrDie() {
  kernel_instance_->set_name(name_);
  kernel_instance_->set_work_item_local_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device_));
  kernel_instance_->set_work_item_private_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device_));

  kernel_instance_->set_outcome(args_set_.Init());
  if (kernel_instance_->outcome() != CldriveKernelInstance::PASS) {
    return;
  }

  for (size_t i = 0; i < instance_->dynamic_params_size(); ++i) {
    *kernel_instance_->add_run() =
        CreateRunForParamsOrDie(instance_->dynamic_params(i),
                                /*output_checks=*/!i);
  }
}

CldriveKernelRun KernelDriver::CreateRunForParamsOrDie(
    const DynamicParams& dynamic_params, const bool output_checks) {
  CldriveKernelRun run;

  // Check that the dynamic params are within legal range.
  auto max_work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (max_work_group_size < dynamic_params.global_size_x()) {
    run.set_outcome(CldriveKernelRun::INVALID_DYNAMIC_PARAMS);
    return run;
  }

  KernelValuesSet inputs;
  args_set_.SetOnes(dynamic_params, &inputs);
  inputs.SetAsArgs(&kernel_);

  KernelValuesSet output_a, output_b;

  *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_a);
  *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_b);

  if (output_a != output_b) {
    run.clear_log();  // Remove performance logs.
    run.set_outcome(CldriveKernelRun::NONDETERMINISTIC);
    return run;
  }

  bool maybe_no_output = output_a == inputs;

  args_set_.SetRandom(dynamic_params, &inputs);
  inputs.SetAsArgs(&kernel_);
  *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_b);

  if (output_a == output_b) {
    run.clear_log();  // Remove performance logs.
    run.set_outcome(CldriveKernelRun::INPUT_INSENSITIVE);
    return run;
  }

  if (maybe_no_output && output_b == inputs) {
    run.clear_log();  // Remove performance logs.
    run.set_outcome(CldriveKernelRun::NO_OUTPUT);
    return run;
  }

  for (size_t i = 3; i < instance_->min_runs_per_kernel(); ++i) {
    *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_a);
  }

  run.set_outcome(CldriveKernelRun::PASS);
  return run;
}

gpu::libcecl::OpenClKernelInvocation KernelDriver::RunOnceOrDie(
    const DynamicParams& dynamic_params, const KernelValuesSet& inputs,
    KernelValuesSet* outputs) {
  LOG(INFO) << "KernelDriver::RunOnceOrDie(" << dynamic_params.local_size_x()
            << "," << dynamic_params.global_size_x() << ")";
  gpu::libcecl::OpenClKernelInvocation log;
  ProfilingData profiling;
  cl::Event event;

  size_t global_size = dynamic_params.global_size_x();
  size_t local_size = dynamic_params.local_size_x();

  log.set_global_size(global_size);
  log.set_local_size(local_size);

  inputs.CopyToDevice(queue_, &profiling);

  queue_.enqueueNDRangeKernel(kernel_, /*offset=*/cl::NullRange,
                              /*global=*/cl::NDRange(global_size),
                              /*local=*/cl::NDRange(local_size),
                              /*events=*/nullptr, /*event=*/&event);
  profiling.elapsed_nanoseconds += GetElapsedNanoseconds(event);

  inputs.CopyFromDeviceToNewValueSet(queue_, outputs, &profiling);

  // Set remained of run proto fields.
  log.set_kernel_name(name_);
  log.set_runtime_ms(profiling.elapsed_nanoseconds / 1000000.0);
  log.set_transferred_bytes(profiling.transferred_bytes);

  return log;
}

}  // namespace cldrive
}  // namespace gpu
