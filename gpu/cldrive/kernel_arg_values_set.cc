#include "gpu/cldrive/kernel_arg_values_set.h"

#include "absl/strings/str_format.h"
#include "phd/logging.h"

namespace gpu {
namespace cldrive {

bool KernelArgValuesSet::operator==(const KernelArgValuesSet &rhs) const {
  CHECK(values_.size() == rhs.values_.size());
  for (size_t i = 0; i < values_.size(); ++i) {
    if (*(values_[i]) != rhs.values_[i].get()) {
      return false;
    }
  }

  return true;
}

bool KernelArgValuesSet::operator!=(const KernelArgValuesSet &rhs) const {
  return !(*this == rhs);
};

void KernelArgValuesSet::CopyToDevice(const cl::CommandQueue &queue,
                                      ProfilingData *profiling) const {
  for (auto &value : values_) {
    value->CopyToDevice(queue, profiling);
  }
}

void KernelArgValuesSet::CopyFromDeviceToNewValueSet(
    const cl::CommandQueue &queue, KernelArgValuesSet *new_values,
    ProfilingData *profiling) const {
  // TODO(cec): Refactor so this isn't causing mallocs() for every run.
  new_values->Clear();
  for (auto &value : values_) {
    new_values->AddKernelArgValue(value->CopyFromDevice(queue, profiling));
  }
}

void KernelArgValuesSet::AddKernelArgValue(
    std::unique_ptr<KernelArgValue> value) {
  values_.push_back(std::move(value));
}

void KernelArgValuesSet::SetAsArgs(cl::Kernel *kernel) {
  for (size_t i = 0; i < values_.size(); ++i) {
    values_[i]->SetAsArg(kernel, i);
  }
}

void KernelArgValuesSet::Clear() { values_.clear(); }

string KernelArgValuesSet::ToString() const {
  string s = "";
  for (size_t i = 0; i < values_.size(); ++i) {
    absl::StrAppend(
        &s, absl::StrFormat("Value[%d] = %s\n", i, values_[i]->ToString()));
  }
  return s;
}

}  // namespace cldrive
}  // namespace gpu
