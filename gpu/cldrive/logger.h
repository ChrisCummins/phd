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
#pragma once

#include "gpu/cldrive/csv_log.h"
#include "gpu/cldrive/proto/cldrive.pb.h"

#include "labm8/cpp/status.h"

#include <iostream>
#include <sstream>

namespace gpu {
namespace cldrive {

// Abstract logging interface for producing consumable output.
class Logger {
 public:
  Logger(std::ostream& ostream, const CldriveInstances* const instances);

  virtual ~Logger() {}

  virtual labm8::Status StartNewInstance();

  // If flush is false, don't emit the log immediately, but instead store the
  // log in a buffer that is emmitted only on a call to PrintAndClearBuffer().
  virtual labm8::Status RecordLog(
      const CldriveInstance* const instance,
      const CldriveKernelInstance* const kernel_instance,
      const CldriveKernelRun* const run,
      const gpu::libcecl::OpenClKernelInvocation* const log, bool flush = true);

  void PrintAndClearBuffer();
  void ClearBuffer();

 protected:
  const CldriveInstances* instances();
  std::ostream& ostream(bool flush);
  int instance_num() const;

 private:
  std::ostream& ostream_;
  std::stringstream buffer_;
  const CldriveInstances* const instances_;
  int instance_num_;
};

// Logging interface for producing protocol buffers.
class ProtocolBufferLogger : public Logger {
 public:
  ProtocolBufferLogger(std::ostream& ostream,
                       const CldriveInstances* const instances,
                       bool text_format);

  virtual ~ProtocolBufferLogger();

 private:
  bool text_format_ = text_format_;
};

class CsvLogger : public Logger {
 public:
  CsvLogger(std::ostream& ostream, const CldriveInstances* const instances);

  virtual labm8::Status RecordLog(
      const CldriveInstance* const instance,
      const CldriveKernelInstance* const kernel_instance,
      const CldriveKernelRun* const run,
      const gpu::libcecl::OpenClKernelInvocation* const log,
      bool flush) override;
};

}  // namespace cldrive
}  // namespace gpu
