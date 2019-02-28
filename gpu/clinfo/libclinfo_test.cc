#include "gpu/clinfo/libclinfo.h"

#include "third_party/opencl/cl.hpp"

#include <vector>
#include "phd/test.h"

namespace phd {
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
}  // namespace phd