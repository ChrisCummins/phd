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
#include "logger.h"

Logger::Logger(std::ostream& ostream, const ClDriveInstances* const instances)
    : ostream_(ostream), instances_(instances), instance_num_(-1) {}

/*virtual*/ phd::Status StartNewInstance() { ++instance_num_; }

/*virtual*/ phd::Status Logger::RecordLog(CldriveInstance* instance) {
  CHECK(instance_num() >= 0);
}

/*virtual*/ phd::Status Logger::End() {
  CHECK(instances_ != nullptr);
  instances_ = nullptr;
}

ClDriveInstances* Logger::instances() { return instances_; }

std::ostream& Logger::ostream() { return ostream_; }

int Logger::instance_num() const { return instance_num_; }

ProtocolBufferLogger::ProtocolBufferLogger(std::ostream& ostream,
                                           bool text_format)
    : Logger(ostream), text_format_(text_format) {}

/*virtual*/ ProtocolBufferLogger::~ProtocolBufferLogger() {
  if (text_format_) {
    ostream() << "# File: //gpu/cldrive/proto/cldrive.proto\n"
              << "# Proto: gpu.cldrive.CldriveInstances\n"
              << instances()->SerializeToString();
  } else {
    ostream() << instances()->DebugString();
  }
}
