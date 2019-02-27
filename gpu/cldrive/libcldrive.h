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

#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

class Cldrive {
 public:
  Cldrive(CldriveInstance* instance, const cl::Device& device);

  void RunOrDie(const bool streaming_csv_output = false);

 private:
  void DoRunOrDie(const bool streaming_csv_output);

  CldriveInstance* instance_;
  cl::Device device_;
};

void ProcessCldriveInstanceOrDie(CldriveInstance* instance);

}  // namespace cldrive
}  // namespace gpu
