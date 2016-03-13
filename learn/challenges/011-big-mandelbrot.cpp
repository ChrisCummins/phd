/*
 * Write a program to compute the mandelbrot set to an arbitrary level
 * of precision (i.e. output to an extremely large image size).
 */
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreserved-id-macro"
#define __CL_ENABLE_EXCEPTIONS
#pragma GCC diagnostic pop

#include <cl.hpp>

#define use_opencl

// output image dimensions:
static const size_t scale = 1000;
static const size_t width = static_cast<size_t>(1.5 * scale),
  height = static_cast<size_t>(1.3 * scale),
  size = width * height;
// mandelbrot range:
static const float start_x = -2.0f, end_x = .75f;
static const float start_y = -1.2f, end_y = 1.2f;
// derived:
static const float dx = (end_x - start_x) / static_cast<float>(width);
static const float dy = (end_y - start_y) / static_cast<float>(height);

// Sequential:
static const unsigned int nmax = 32;

// divide image into blocks:
static const size_t bsize = 1024 * 1024;

#ifdef use_opencl  // OpenCL kernel
const char* mandelbrot_kernel = R"(
__kernel void mandelbrot(__global unsigned char* out,
                         const float start_x, const float start_y,
                         const float dx, const float dy,
                         const unsigned long width,
                         const unsigned long offset,
                         const unsigned long size,
                         const unsigned int nmax) {
  const unsigned int i = get_global_id(0);
  if (offset + i < size) {
    int n = 0;
    float r = 0, s = 0, next = 0;
    const unsigned int local_x = (offset + i) % width;
    const unsigned int local_y = (offset + i) / width;
    const float x = start_x + local_x * dx;
    const float y = start_y + local_y * dy;

    while (r * r + s * s <= 4 && n < nmax) {
      next = r * r - s * s + x;
      s = 2 * r * s + y;
      r = next;
      ++n;
    }

    const float scaled = max(n / (float)nmax, 0.1);
    out[i * 3    ] = scaled == 1.0 ? 0u : scaled * scaled * 255u;
    out[i * 3 + 1] = scaled == 1.0 ? 0u : scaled * 128u;
    out[i * 3 + 2] = scaled == 1.0 ? 0u : scaled * 255u;
  }
})";
#endif

int main() {
  try {
    std::cout << "image size " << width << " x " << height << "px" << std::endl;

    auto io_time = 0.0;
    unsigned char buf[bsize * 3];

    std::ofstream file;
    file.open("011-big-mandelbrot.ppm");
    file << "P3" << std::endl;
    file << width << " " << height << std::endl;
    file << "255" << std::endl;

#ifdef use_opencl
    // Setup OpenCL:
    std::cout << "using opencl ..." << std::endl;

    // Create a context
    cl::Context context{CL_DEVICE_TYPE_DEFAULT};
    cl::Program program(context, mandelbrot_kernel, true);
    cl::CommandQueue queue(context);
    auto kern = cl::make_kernel<cl::Buffer,
                                float, float, float, float,
                                size_t, size_t, size_t,
                                unsigned int>(program, "mandelbrot");
    auto out = cl::Buffer(context, std::begin(buf), std::end(buf), true);
#endif

    auto timer = std::clock();

    for (size_t i = 0; i < size; i += bsize) {
      // print
      std::cout << " block " << i / bsize + 1 << " of "
                << std::ceil(size / bsize) + 1
                << ", " << std::min(bsize, size - i)
                << " pixels" << std::endl;

#ifdef use_opencl
      // OpenCL:
      kern(cl::EnqueueArgs(queue, cl::NDRange(bsize)),
           out, start_x, start_y, dx, dy, width, i, size, nmax);

      queue.finish();
      cl::copy(queue, out, std::begin(buf), std::end(buf));
#else
      for (size_t j = 0; j < bsize; j++) {
        const size_t global_id = j + i;

        if (global_id < size) {
          auto n = 0u;
          auto r = 0.0f, s = 0.0f, next = 0.0f;
          const auto local_x = global_id % width;
          const auto local_y = global_id / width;
          const auto x = start_x + local_x * dx, y = start_y + local_y * dy;

          while (r * r + s * s <= 4 && n < nmax) {
            next = r * r - s * s + x;
            s = 2 * r * s + y;
            r = next;
            ++n;
          }

          // Convert z to colour:
          const float scaled = std::max(n / static_cast<float>(nmax), 0.1f);

          if (scaled < 1.0) {
            buf[j * 3    ] = static_cast<unsigned char>(scaled * scaled * 255u);
            buf[j * 3 + 1] = static_cast<unsigned char>(scaled * 128u);
            buf[j * 3 + 2] = static_cast<unsigned char>(scaled * 255u);
          } else {
            buf[j * 3] = buf[j * 3 + 1] = buf[j * 3 + 2] = 0u;
          }
        }
      }
#endif

      // Write block to file
      const auto io_begin = std::clock();
      unsigned char *it = buf;
      while (it != buf + std::min(bsize, size - i) * 3) {
        file << static_cast<unsigned int>(*it++) << ' ';
      }
      io_time += (std::clock() - io_begin)
                 / static_cast<double>(CLOCKS_PER_SEC);
    }

    auto duration = (std::clock() - timer)
                    / static_cast<double>(CLOCKS_PER_SEC);
    std::cout << "total = "<< duration << "s, "
              << "io = " << io_time << "s, "
              << "remaining = " << duration - io_time << "s" << std::endl;

    file.close();
  } catch (cl::Error&) {
    std::cerr << "woops!\n";
  }

  return 0;
}
