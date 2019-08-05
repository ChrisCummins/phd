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
#include "gpu/cldrive/logger.h"
#include "labm8/cpp/logging.h"

namespace gpu {
namespace cldrive {

Logger::Logger(std::ostream& ostream, const CldriveInstances* const instances)
    : ostream_(ostream), instances_(instances), instance_num_(-1) {}

/*virtual*/ labm8::Status Logger::StartNewInstance() {
  ++instance_num_;
  return labm8::Status::OK;
}

/*virtual*/ labm8::Status Logger::RecordLog(
    const CldriveInstance* const instance,
    const CldriveKernelInstance* const kernel_instance,
    const CldriveKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log, bool flush) {
  CHECK(instance_num() >= 0);
  return labm8::Status::OK;
}

void Logger::PrintAndClearBuffer() {
  ostream_ << buffer_.str();
  ClearBuffer();
}

void Logger::ClearBuffer() {
  buffer_.clear();
  buffer_.str(string());
}

const CldriveInstances* Logger::instances() { return instances_; }

std::ostream& Logger::ostream(bool flush) {
  if (flush) {
    return ostream_;
  } else {
    return buffer_;
  }
}

int Logger::instance_num() const { return instance_num_; }

ProtocolBufferLogger::ProtocolBufferLogger(
    std::ostream& ostream, const CldriveInstances* const instances,
    bool text_format)
    : Logger(ostream, instances), text_format_(text_format) {}

/*virtual*/ ProtocolBufferLogger::~ProtocolBufferLogger() {
  if (text_format_) {
    ostream(/*flush=*/true) << "# File: //gpu/cldrive/proto/cldrive.proto\n"
                            << "# Proto: gpu.cldrive.CldriveInstances\n"
                            << instances()->DebugString();
  } else {
    instances()->SerializeToOstream(&ostream(/*flush=*/true));
  }
}

CsvLogger::CsvLogger(std::ostream& ostream,
                     const CldriveInstances* const instances)
    : Logger(ostream, instances) {
  this->ostream(/*flush=*/true) << CsvLogHeader();
}

/*virtual*/ labm8::Status CsvLogger::RecordLog(
    const CldriveInstance* const instance,
    const CldriveKernelInstance* const kernel_instance,
    const CldriveKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log, bool flush) {
  ostream(flush) << CsvLog::FromProtos(instance_num(), instance,
                                       kernel_instance, run, log);
  return labm8::Status::OK;
}

}  // namespace cldrive
}  // namespace gpu
