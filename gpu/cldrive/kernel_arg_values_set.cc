// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/kernel_arg_values_set.h"

#include "absl/strings/str_format.h"
#include "labm8/cpp/logging.h"

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
  for (auto &value : values()) {
    value->CopyToDevice(queue, profiling);
  }
}

void KernelArgValuesSet::CopyFromDeviceToNewValueSet(
    const cl::CommandQueue &queue, KernelArgValuesSet *new_values,
    ProfilingData *profiling) const {
  // TODO(cec): Refactor so this isn't causing mallocs() for every run.
  new_values->Clear();
  for (auto &value : values()) {
    new_values->AddKernelArgValue(value->CopyFromDevice(queue, profiling));
  }
}

void KernelArgValuesSet::AddKernelArgValue(
    std::unique_ptr<KernelArgValue> value) {
  values().push_back(std::move(value));
}

void KernelArgValuesSet::SetAsArgs(cl::Kernel *kernel) {
  for (size_t i = 0; i < values().size(); ++i) {
    values()[i]->SetAsArg(kernel, i);
  }
}

void KernelArgValuesSet::Clear() { values().clear(); }

string KernelArgValuesSet::ToString() const {
  string s = "";
  for (size_t i = 0; i < values().size(); ++i) {
    absl::StrAppend(
        &s, absl::StrFormat("Arg[%d]: %s\n", i + 1, values()[i]->ToString()));
  }
  return s;
}

std::vector<std::unique_ptr<KernelArgValue>> &KernelArgValuesSet::values() {
  return values_;
}

const std::vector<std::unique_ptr<KernelArgValue>> &KernelArgValuesSet::values()
    const {
  return values_;
}

}  // namespace cldrive
}  // namespace gpu
