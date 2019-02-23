#include "gpu/cldrive/kernel_values_set.h"

#include "absl/strings/str_format.h"

namespace gpu {
namespace cldrive {

bool KernelValuesSet::operator==(const KernelValuesSet &rhs) const {
  CHECK(values_.size() == rhs.values_.size());
  for (size_t i = 0; i < values_.size(); ++i) {
    if (*(values_[i]) != rhs.values_[i].get()) {
      // LOG(DEBUG) << "Kernel value " << i << " not equal";
      return false;
    }
  }

  return true;
}

bool KernelValuesSet::operator!=(const KernelValuesSet &rhs) const {
  return !(*this == rhs);
};

void KernelValuesSet::CopyToDevice(const cl::CommandQueue &queue,
                                   ProfilingData *profiling) const {
  for (auto &value : values_) {
    value->CopyToDevice(queue, profiling);
  }
}

void KernelValuesSet::CopyFromDeviceToNewValueSet(
    const cl::CommandQueue &queue, KernelValuesSet *new_values,
    ProfilingData *profiling) const {
  // TODO(cec): Refactor so this isn't causing mallocs() for every run.
  new_values->Clear();
  for (auto &value : values_) {
    new_values->AddKernelValue(value->CopyFromDevice(queue, profiling));
  }
}

void KernelValuesSet::AddKernelValue(std::unique_ptr<KernelValue> value) {
  values_.push_back(std::move(value));
}

void KernelValuesSet::SetAsArgs(cl::Kernel *kernel) {
  for (size_t i = 0; i < values_.size(); ++i) {
    values_[i]->SetAsArg(kernel, i);
  }
}

void KernelValuesSet::Clear() { values_.clear(); }

string KernelValuesSet::ToString() const {
  string s = "";
  for (size_t i = 0; i < values_.size(); ++i) {
    absl::StrAppend(
        &s, absl::StrFormat("Value[%d] = %s\n", i, values_[i]->ToString()));
  }
  return s;
}

}  // namespace cldrive
}  // namespace gpu
