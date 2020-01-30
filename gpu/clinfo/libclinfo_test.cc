// Copyright 2017-2020 Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "gpu/clinfo/libclinfo.h"

#include "labm8/cpp/test.h"
#include "third_party/opencl/cl.hpp"

#include <vector>

namespace labm8 {
namespace gpu {
namespace clinfo {
namespace {

TEST(GetDefaultOpenClDeviceOrDie, CreateABufferUsingDefaultDevice) {
  cl::Device device = GetDefaultOpenClDeviceOrDie();
  cl::Context context(device);
  cl::CommandQueue queue(context,
                         /*devices=*/context.getInfo<CL_CONTEXT_DEVICES>()[0],
                         /*properties=*/CL_QUEUE_PROFILING_ENABLE);

  // Create an OpenCL buffer of 10 ints.
  size_t buffer_size = sizeof(int) * 10;
  cl::Buffer buffer(context, /*flags=*/CL_MEM_READ_WRITE,
                    /*size=*/buffer_size);

  // Map the buffer to a pointer.
  cl::Event event;
  int *pointer = static_cast<int *>(queue.enqueueMapBuffer(
      buffer, /*blocking=*/true, /*flags=*/CL_MAP_WRITE, /*offset=*/0,
      /*size=*/buffer_size, /*events=*/nullptr, /*event=*/&event,
      /*error=*/nullptr));
  event.wait();

  EXPECT_NE(pointer, nullptr);

  // Unmap the buffer.
  queue.enqueueUnmapMemObject(buffer, pointer, /*events=*/nullptr, &event);
  event.wait();
}

}  // anonymous namespace
}  // namespace clinfo
}  // namespace gpu
}  // namespace labm8

TEST_MAIN();
