// TODO(cec): A VERY work-in-progress implementation of cldrive in C++.

#include "gpu/cldrive/native_driver_lib.h"

#include <cstdlib>

#include "third_party/opencl/include/cl.hpp"

#include "phd/logging.h"
#include "phd/macros.h"
#include "phd/status.h"
#include "phd/statusor.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"

#define LOG_CL_ERROR(level, error)                                             \
  LOG(level) << "OpenCL exception: " << error.what() << ", error: "            \
             << phd::gpu::clinfo::OpenClErrorString(error.err());

namespace gpu {
namespace cldrive {

namespace {

// Lookup an OpenCL device by proto or die.
cl::Device GetOpenClDeviceOrDie(const ::gpu::clinfo::OpenClDevice &device) {
  try {
    return phd::gpu::clinfo::GetOpenClDevice(device);
  } catch (std::invalid_argument e) {
    LOG(FATAL) << "Could not find device '" << device.name() << "'";
  }
}

// Attempt to build OpenCL program.
phd::StatusOr<cl::Program>
BuildOpenClProgram(const std::string &opencl_kernel,
                   const std::vector<cl::Device> &devices) {
  try {
    cl::Program program(opencl_kernel);
    program.build(devices);
    return program;
  } catch (cl::Error e) {
    LOG_CL_ERROR(ERROR, e);
    return phd::Status::UNKNOWN;
  }
}

class ProfilingData {
public:
  ProfilingData() : elapsed_nanoseconds(0), transferred_bytes(0) {}
  phd::int64 elapsed_nanoseconds;
  phd::int64 transferred_bytes;
};

phd::int64 GetElapsedNanoseconds(cl::Event event) {
  event.wait();
  cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
  cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return end - start;
}

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
template <typename IteratorType>
void CopyHostToDevice(const cl::CommandQueue &queue, IteratorType startIterator,
                      IteratorType endIterator, const cl::Buffer &buffer,
                      ProfilingData *profiling) {
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;
  size_t length = endIterator - startIterator;
  size_t byteLength = length * sizeof(ValueType);

  cl::Event event;
  ValueType *pointer = static_cast<ValueType *>(queue.enqueueMapBuffer(
      buffer, /*blocking=*/true, /*flags=*/CL_MAP_WRITE, /*offset=*/0,
      /*size=*/byteLength, /*events=*/nullptr, /*event=*/&event,
      /*error=*/nullptr));
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  std::copy(startIterator, endIterator, pointer);

  queue.enqueueUnmapMemObject(buffer, pointer, /*events=*/nullptr, &event);

  // Set profiling data.
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += byteLength;
}

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
template <typename IteratorType>
void CopyDeviceToHost(const cl::CommandQueue &queue, const cl::Buffer &buffer,
                      IteratorType startIterator,
                      const IteratorType endIterator,
                      ProfilingData *profiling) {
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;
  size_t length = endIterator - startIterator;
  size_t byteLength = length * sizeof(ValueType);

  cl::Event event;
  ValueType *pointer = static_cast<ValueType *>(queue.enqueueMapBuffer(
      buffer, /*blocking=*/true, /*flags=*/CL_MAP_READ, /*offset=*/0,
      /*size=*/byteLength, /*events=*/nullptr, /*event=*/&event,
      /*error=*/nullptr));
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  std::copy(pointer, pointer + length, startIterator);

  queue.enqueueUnmapMemObject(buffer, pointer, /*events=*/nullptr, &event);

  // Set profiling data.
  profiling->elapsed_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += byteLength;
}

class KernelValue {
public:
  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) = 0;

  virtual std::unique_ptr<KernelValue>
  CopyFromDevice(const cl::CommandQueue &queue, ProfilingData *profiling) = 0;

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) = 0;

  virtual bool operator==(const KernelValue *const rhs) const = 0;

  virtual bool operator!=(const KernelValue *const rhs) const = 0;

  virtual string ToString() const = 0;
};

template <typename T> class ArrayValue : public KernelValue {
public:
  template <typename... Args>
  ArrayValue(size_t size, Args &&... args) : vector_(size, args...) {}

  virtual bool operator==(const KernelValue *const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ArrayValue *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    if (vector().size() != rhs_ptr->vector().size()) {
//      LOG(DEBUG) << "Array size not equal (" << vector().size()
//                 << " != " << rhs_ptr->vector().size() << ")";
      return false;
    }

    for (size_t i = 0; i < vector().size(); ++i) {
      if (vector()[i] != rhs_ptr->vector()[i]) {
//        LOG(DEBUG) << "Array element " << i << " not equal (" << vector()[i]
//                   << " != " << rhs_ptr->vector()[i] << ")";
        return false;
      }
    }

    return true;
  }

  virtual bool operator!=(const KernelValue *const rhs) const override {
    return !(*this == rhs);
  }

  std::vector<T> &vector() { return vector_; }

  const std::vector<T> &vector() const { return vector_; }

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override {
    CHECK(false);
  }

  virtual std::unique_ptr<KernelValue>
  CopyFromDevice(const cl::CommandQueue &queue,
                 ProfilingData *profiling) override {
    CHECK(false);
  }

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    CHECK(false);
  }

  virtual string ToString() const override {
    string s = "";
    for (auto& value : vector()) {
      absl::StrAppend(&s, value);
      absl::StrAppend(&s, ", ");
    }
    return s;
  }

protected:
  std::vector<T> vector_;
};

template <typename T> class ArrayValueWithBuffer : public ArrayValue<T> {
public:
  template <typename... Args>
  ArrayValueWithBuffer(const cl::Context &context, size_t size, Args &&... args)
      : ArrayValue<T>(size, args...),
        buffer_(context, /*flags=*/CL_MEM_READ_WRITE,
                /*size=*/sizeof(T) * size) {}

  cl::Buffer &buffer() { return buffer_; }

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, buffer());
  }

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override {
    CopyHostToDevice(queue, this->vector().begin(), this->vector().end(),
                     buffer(), profiling);
  }

  virtual std::unique_ptr<KernelValue>
  CopyFromDevice(const cl::CommandQueue &queue,
                 ProfilingData *profiling) override {
    auto new_arg = std::make_unique<ArrayValue<T>>(this->vector().size());
    CopyDeviceToHost(queue, buffer(), new_arg->vector().begin(),
                     new_arg->vector().end(), profiling);
    return std::move(new_arg);
  }

private:
  cl::Buffer buffer_;
};

template <typename T> class ScalarKernelArg : public KernelValue {
public:
  ScalarKernelArg(const T &value) : value_(value) {}

  virtual bool operator==(const KernelValue*const rhs) const override {
    auto rhs_ptr = dynamic_cast<const ScalarKernelArg *const>(rhs);
    if (!rhs_ptr) {
      return false;
    }

    return value() == rhs_ptr->value();
  }

  virtual bool operator!=(const KernelValue *const rhs) const override {
    return !(*this == rhs);
  };

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) override {
    kernel->setArg(arg_index, value());
  };

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) override{};

  virtual std::unique_ptr<KernelValue>
  CopyFromDevice(const cl::CommandQueue &queue,
                 ProfilingData *profiling) override {
    return std::make_unique<ScalarKernelArg>(value());
  }

  const T &value() const { return value_; }
  T &value() { return value_; }

  virtual string ToString() const override {
    return absl::StrFormat("%d", value_);
  }

private:
  T value_;
};

class KernelValuesSet {
public:
  bool operator==(const KernelValuesSet &rhs) const {
    CHECK(values_.size() == rhs.values_.size());
    for (size_t i = 0; i < values_.size(); ++i) {
      if (*(values_[i]) != rhs.values_[i].get()) {
        // LOG(DEBUG) << "Kernel value " << i << " not equal";
        return false;
      }
    }

    return true;
  }

  bool operator!=(const KernelValuesSet &rhs) const {
    return !(*this == rhs);
  };

  void CopyToDevice(const cl::CommandQueue &queue,
                    ProfilingData *profiling) const {
    for (auto &value : values_) {
      value->CopyToDevice(queue, profiling);
    }
  }

  void CopyFromDeviceToNewValueSet(const cl::CommandQueue &queue,
                                   KernelValuesSet *new_values,
                                   ProfilingData *profiling) const {
    // TODO(cec): Refactor so this isn't causing mallocs() for every run.
    new_values->Clear();
    for (auto &value : values_) {
      new_values->AddKernelValue(value->CopyFromDevice(queue, profiling));
    }
  }

  void AddKernelValue(std::unique_ptr<KernelValue> value) {
    values_.push_back(std::move(value));
  }

  void SetAsArgs(cl::Kernel *kernel) {
    for (size_t i = 0; i < values_.size(); ++i) {
      values_[i]->SetAsArg(kernel, i);
    }
  }

  void Clear() { values_.clear(); }

  string ToString() const {
    string s = "";
    for (size_t i = 0; i < values_.size(); ++i) {
      absl::StrAppend(&s, absl::StrFormat(
          "Value[%d] = %s\n", i, values_[i]->ToString()));
    }
    return s;
  }

private:
  std::vector<std::unique_ptr<KernelValue>> values_;
};

// template<typename T>
class KernelArgDriver {
public:
  KernelArgDriver(const cl::Context &context, cl::Kernel *kernel,
                  size_t arg_index)
      : context_(context), kernel_(kernel), arg_index_(arg_index),
        address_(
            kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index)) {

    string full_type_name =
        kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
    CHECK(full_type_name.size());

    // Strip the final '*' character form the type name, if present.
    is_pointer_ = full_type_name[full_type_name.size() - 2] == '*';
    if (is_pointer_) {
      type_name_ = full_type_name.substr(full_type_name.size() - 2);
    } else {
      type_name_ = full_type_name;
    }
  }

  phd::Status Init() {
    // Address qualifier is one of:
    //   CL_KERNEL_ARG_ACCESS_READ_ONLY
    //   CL_KERNEL_ARG_ACCESS_WRITE_ONLY
    //   CL_KERNEL_ARG_ACCESS_READ_WRITE
    //   CL_KERNEL_ARG_ACCESS_NONE
    //
    // If argument is not an image type, CL_KERNEL_ARG_ACCESS_NONE is returned.
    // If argument is an image type, the access qualifier specified or the
    // default access qualifier is returned.
    auto access_qualifier =
        kernel_->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index_);
    if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
      LOG(ERROR) << "Argument " << arg_index_ << " is an image type";
      return phd::Status::UNKNOWN;
    }

    return phd::Status::OK;
  }

  std::unique_ptr<KernelValue>
  CreateRandom(const DynamicParams &dynamic_params) {
    if (IsMutable()) {
      auto arg_buffer = std::make_unique<ArrayValueWithBuffer<int>>(
          context_, dynamic_params.global_size_x());
      for (size_t i = 0; i < dynamic_params.global_size_x(); ++i) {
        arg_buffer->vector()[i] = rand();
      }
      // TODO(cec): Populate with random values.
      return std::move(arg_buffer);
    } else {
      return std::make_unique<ScalarKernelArg<int>>(
          dynamic_params.global_size_x());
    }
  }

  std::unique_ptr<KernelValue> CreateOnes(const DynamicParams &dynamic_params) {
    if (IsMutable()) {
      auto arg_buffer = std::make_unique<ArrayValueWithBuffer<int>>(
          context_, dynamic_params.global_size_x(), 1);
      return std::move(arg_buffer);
    } else {
      return std::make_unique<ScalarKernelArg<int>>(
          dynamic_params.global_size_x());
    }
  }

  bool IsMutable() const { return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL; }

  bool IsPointer() const { return is_pointer_; }

private:
  cl::Context context_;
  cl::Kernel *kernel_;
  size_t arg_index_;

  // One of:
  //   CL_KERNEL_ARG_ADDRESS_GLOBAL
  //   CL_KERNEL_ARG_ADDRESS_LOCAL
  //   CL_KERNEL_ARG_ADDRESS_CONSTANT
  //   CL_KERNEL_ARG_ADDRESS_PRIVATE
  cl_kernel_arg_address_qualifier address_;
  string type_name_;
  bool is_pointer_;
};

class KernelArgSet {
public:
  KernelArgSet(const cl::Context &context, cl::Kernel *kernel)
      : context_(context), kernel_(kernel) {}

  CldriveKernelInstance::KernelInstanceOutcome
  LogErrorOutcome(const CldriveKernelInstance::KernelInstanceOutcome &outcome) {
    LOG(ERROR) << "Kernel " << kernel_->getInfo<CL_KERNEL_FUNCTION_NAME>()
               << " "
               << CldriveKernelInstance::KernelInstanceOutcome_Name(outcome);
    return outcome;
  }

  CldriveKernelInstance::KernelInstanceOutcome Init() {
    // Early exit if the kernel has no arguments.
    size_t num_args = kernel_->getInfo<CL_KERNEL_NUM_ARGS>();
    if (!num_args) {
      return LogErrorOutcome(CldriveKernelInstance::NO_ARGUMENTS);
    }

    // Create args.
    int num_mutable_args = 0;
    for (size_t i = 0; i < num_args; ++i) {
      auto arg_driver = KernelArgDriver(context_, kernel_, i);
      if (!arg_driver.Init().ok()) {
        // Early exit if argument is not supported.
        return LogErrorOutcome(CldriveKernelInstance::UNSUPPORTED_ARGUMENTS);
      }
      if (arg_driver.IsMutable()) {
        ++num_mutable_args;
      }
      args_.push_back(std::move(arg_driver));
    }

    // Early exit if the kernel has no mutable arguments.
    if (!num_mutable_args) {
      return LogErrorOutcome(CldriveKernelInstance::NO_MUTABLE_ARGUMENTS);
    }

    return CldriveKernelInstance::PASS;
  }

  void SetRandom(const DynamicParams &dynamic_params, KernelValuesSet *values) {
    // LOG(INFO) << "SetRandom()";
    values->Clear();
    for (auto &arg : args_) {
      values->AddKernelValue(arg.CreateRandom(dynamic_params));
    }
  }

  void SetOnes(const DynamicParams &dynamic_params, KernelValuesSet *values) {
    // LOG(INFO) << "SetOnes()";
    values->Clear();
    for (auto &arg : args_) {
      values->AddKernelValue(arg.CreateOnes(dynamic_params));
    }
  }

private:
  cl::Context context_;
  cl::Kernel *kernel_;
  std::vector<KernelArgDriver> args_;
};

class KernelDriver {
public:
  KernelDriver(const cl::Context &context, const cl::CommandQueue &queue,
               const cl::Kernel &kernel, CldriveInstance *instance)
      : context_(context), queue_(queue),
        device_(context.getInfo<CL_CONTEXT_DEVICES>()[0]), kernel_(kernel),
        instance_(instance), kernel_instance_(instance->add_kernel()),
        name_(kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()),
        args_set_(context, &kernel_) {}

  void RunOrDie() {
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

  CldriveKernelRun CreateRunForParamsOrDie(const DynamicParams &dynamic_params,
                                           const bool output_checks) {
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
      run.clear_log(); // Remove performance logs.
      run.set_outcome(CldriveKernelRun::NONDETERMINISTIC);
      return run;
    }

    bool maybe_no_output = output_a == inputs;

    args_set_.SetRandom(dynamic_params, &inputs);
    inputs.SetAsArgs(&kernel_);
    *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_b);

    if (output_a == output_b) {
      run.clear_log(); // Remove performance logs.
      run.set_outcome(CldriveKernelRun::INPUT_INSENSITIVE);
      return run;
    }

    if (maybe_no_output && output_b == inputs) {
      run.clear_log(); // Remove performance logs.
      run.set_outcome(CldriveKernelRun::NO_OUTPUT);
      return run;
    }

    for (size_t i = 3; i < instance_->min_runs_per_kernel(); ++i) {
      *run.add_log() = RunOnceOrDie(dynamic_params, inputs, &output_a);
    }

    run.set_outcome(CldriveKernelRun::PASS);
    return run;
  }

  gpu::libcecl::OpenClKernelInvocation
  RunOnceOrDie(const DynamicParams &dynamic_params,
               const KernelValuesSet &inputs, KernelValuesSet *outputs) {
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

private:
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Device device_;
  cl::Kernel kernel_;
  CldriveInstance *instance_;
  CldriveKernelInstance *kernel_instance_;
  string name_;
  KernelArgSet args_set_;
};

class Cldrive {
public:
  explicit Cldrive(CldriveInstance *instance)
      : instance_(instance), device_(GetOpenClDeviceOrDie(instance->device())),
        context_(device_),
        queue_(context_, context_.getInfo<CL_CONTEXT_DEVICES>()[0],
               /*properties=*/CL_QUEUE_PROFILING_ENABLE) {}

  void RunOrDie() {
    // Compile program or fail.
    phd::StatusOr<cl::Program> program_or =
        BuildOpenClProgram(string(instance_->opencl_src()),
                           context_.getInfo<CL_CONTEXT_DEVICES>());
    if (!program_or.ok()) {
      LOG(ERROR) << "OpenCL program compilation failed!";
      instance_->set_outcome(CldriveInstance::PROGRAM_COMPILATION_FAILURE);
      return;
    }
    cl::Program program = program_or.ValueOrDie();

    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);

    if (!kernels.size()) {
      LOG(ERROR) << "OpenCL program contains no kernels!";
      instance_->set_outcome(CldriveInstance::NO_KERNELS_IN_PROGRAM);
      return;
    }

    for (auto &kernel : kernels) {
      KernelDriver(context_, queue_, kernel, instance_).RunOrDie();
    }

    instance_->set_outcome(CldriveInstance::PASS);
  };

private:
  CldriveInstance *instance_;
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
};

} // namespace

void ProcessCldriveInstanceOrDie(CldriveInstance *instance) {
  try {
    Cldrive(instance).RunOrDie();
  } catch (cl::Error error) {
    LOG_CL_ERROR(FATAL, error);
  }
}

} // namepsace cldrive
} // namepsace gpu
