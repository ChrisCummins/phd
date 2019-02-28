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

#include "phd/status.h"

#include "gpu/cldrive/proto/cldrive.pb.h"

#include <iostream>

// Abstract logging interface for producing consumable output.
class Logger {
 public:
  Logger(std::ostream& ostream, const CldriveInstances* const instances);

  virtual ~Logger() {}

  virtual phd::Status StartNewInstance();

  virtual phd::Status RecordLog(
      const CldriveInstance* const instance,
      const CldriveKernelInstance* const kernel_instance,
      const CldriveKernelRun* const run,
      const gpu::libcecl::OpenClKernelInvocation* const log);

 protected:
  CldriveInstances* instances();
  std::ostream& ostream();
  int instance_num() const;

 private:
  std::ostream& ostream_;
  const CldriveInstances* const instances_;
  int instance_num_;
};

// Logging interface for producing protocol buffers.
class ProtocolBufferLogger : public Logger {
 public:
  ProtocolBufferLogger(std::ostream& ostream, bool text_format);

  virtual ~ProtocolBufferLogger();

  virtual phd::Status End() override;

 private:
  bool text_format_ = text_format_;
};
