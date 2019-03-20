// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
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
#pragma once

#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/port.h"
#include "phd/string.h"

#include <vector>

namespace gpu {
namespace cldrive {

// A class which prints the header values for a CSV row.
//
// Usage:
//    std::cout << CsvLogHeader();
class CsvLogHeader {
  // Format CSV header to output stream.
  friend std::ostream& operator<<(std::ostream& stream,
                                  const CsvLogHeader& log);
};

// A class which formats a CSV row.
//
// Usage:
//    CsvLog::FromProtos log(...);
//    std::cout << log;
class CsvLog {
 public:
  CsvLog(int instance_id);

  // Create a log from proto messages.
  static CsvLog FromProtos(
      int instance_id, const CldriveInstance* const instance,
      const CldriveKernelInstance* const kernel_instance,
      const CldriveKernelRun* const run,
      const gpu::libcecl::OpenClKernelInvocation* const log);

  // Format CSV to output stream.
  friend std::ostream& operator<<(std::ostream& stream, const CsvLog& log);

 private:
  // Begin CSV columns (in order) -----------------------------------

  // From CldriveInstance.opencl_src field.
  int instance_id_;

  // From CldriveInstance.device.name field.
  string device_;

  // From CldriveInstance.build_opts field.
  string build_opts_;

  // The OpenCL kernel name, from CldriveKernelInstance.name
  // field. If CldriveInstance.outcome != PASS, this will be empty.
  string kernel_;

  // From CldriveKernelInstance.work_item_local_mem_size_in_bytes field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int work_item_local_mem_size_;

  // From CldriveKernelInstance.work_item_private_mem_size_in_bytes field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int work_item_private_mem_size_;

  // From CldriveInstance.dynamic_params.global_size_x field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int global_size_;

  // From CldriveInstance.dynamic_params.local_size_x field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int local_size_;

  // A stringified enum value. Either CldriveInstance.outcome if
  // CldriveInstance.outcome != PASS, else CldriveKernelInstance.outcome if
  // CldriveKernelInstance.outcome != PASS, else CldriveKernelRun.outcome.
  string outcome_;

  // From CldriveKernelRun.log.transferred_bytes. If outcome != PASS, this will
  // be empty.
  phd::int64 transferred_bytes_;

  // From CldriveKernelRun.log.runtime_ms. If outcome != PASS, this will be
  // empty.
  double runtime_ms_;

  // End CSV columns (in order) -----------------------------------
};

//
std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& log);
std::ostream& operator<<(std::ostream& stream, const CsvLog& log);

}  // namespace cldrive
}  // namespace gpu
