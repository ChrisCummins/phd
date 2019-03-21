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

#include "phd/port.h"

#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace cldrive {

phd::int64 GetElapsedNanoseconds(const cl::Event& event);

class ProfilingData {
 public:
  ProfilingData()
      : kernel_nanoseconds(0), transfer_nanoseconds(0), transferred_bytes(0) {}
  phd::int64 kernel_nanoseconds;
  phd::int64 transfer_nanoseconds;
  phd::int64 transferred_bytes;
};

}  // namespace cldrive
}  // namespace gpu
