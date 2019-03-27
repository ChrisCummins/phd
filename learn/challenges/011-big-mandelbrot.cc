/*
 * Write a program to compute the mandelbrot set to an arbitrary level
 * of precision (i.e. output to an extremely large image size).
 */

// Configuration:
//
// use_opencl - if defined, use opencl to compute pixel values. Else,
//              use sequential CPU.
// use_mmap - if defined, map output file to memory for writing. Else,
//            use output file stream.
#define use_mmap

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef use_mmap
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#endif  // use_mmap

#ifdef use_opencl
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreserved-id-macro"
#define __CL_ENABLE_EXCEPTIONS
#pragma GCC diagnostic pop
#include "third_party/opencl/cl.hpp"
#endif  // use_opencl

// output image dimensions:
static const size_t scale = 5000;
static const auto width = static_cast<size_t>(1.5 * scale),
                  height = static_cast<size_t>(1.3 * scale),
                  size = width * height;

// mandelbrot parameters:
static const float start_x = -2.0f, end_x = .75f;
static const float start_y = -1.2f, end_y = 1.2f;
static const unsigned int nmax = 32;
// derived:
static const float dx = (end_x - start_x) / static_cast<float>(width);
static const float dy = (end_y - start_y) / static_cast<float>(height);

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
    float r = 0, s = 0, next = 0;
    const unsigned int local_x = (offset + i) % width;
    const unsigned int local_y = (offset + i) / width;
    const float x = start_x + local_x * dx;
    const float y = start_y + local_y * dy;
    int n;

    for (n = 0; n < nmax; ++n) {
      if (r * r + s * s > 4.0)
        break;
      next = r * r - s * s + x;
      s = 2 * r * s + y;
      r = next;
    }

    const float scaled = max(n / (float)nmax, 0.1);
    out[i * 3    ] = scaled == 1.0 ? 0u : scaled * scaled * 255u;
    out[i * 3 + 1] = scaled == 1.0 ? 0u : scaled * 128u;
    out[i * 3 + 2] = scaled == 1.0 ? 0u : scaled * 255u;
  }
})";
#endif  // use_opencl

template <typename T, typename Y = int,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
auto ndigits(T number) {
  Y digits = 0;
  while (number) {
    number /= 10;
    ++digits;
  }
  return digits;
}

int main() {
  int fd{-1}, ret{0};

  try {
    auto io_time = 0.0;
    unsigned char buf[bsize * 3];

    // File header:
    std::ostringstream header;
    header << "P6\n" << width << ' ' << height << '\n' << "255\n";

    // Output file:
#ifdef use_mmap
    const auto headerlen = strlen(header.str().c_str());
    const auto filesize = headerlen + size * 3 * sizeof(char);
    fd = open("011-big-mandelbrot.ppm", O_RDWR | O_CREAT | O_TRUNC,
              static_cast<mode_t>(0600));
    if (fd == -1) throw std::runtime_error{"couldn't open file"};
    if (lseek(fd, static_cast<off_t>(filesize - 1u), SEEK_SET) == -1)
      throw std::runtime_error{"couldn't stretch file file"};
    if (write(fd, "", 1) == -1)
      throw std::runtime_error{"couldn't write last byte of file"};
    char* map = static_cast<char*>(
        mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (map == MAP_FAILED) throw std::runtime_error{"couldn't map file"};
    char* f = map;

    // print stuff
    std::cout << "image size " << width << " x " << height << " px, "
              << "file size: " << filesize / 1024.0 / 1024 << " Mb" << std::endl
              << std::endl;

    // Write file header
    const char* h = header.str().c_str();
    while (*h) *f++ = *h++;
#else   // ndef use_mmap
    std::ofstream file;
    file.open("011-big-mandelbrot.ppm");
    file << header.str();
#endif  // use_mmap

#ifdef use_opencl
    // Setup OpenCL:
    std::cout << "using opencl ..." << std::endl;

    // Create a context
    cl::Context context{CL_DEVICE_TYPE_DEFAULT};
    cl::Program program(context, mandelbrot_kernel, true);
    cl::CommandQueue queue(context);
    auto kern =
        cl::make_kernel<cl::Buffer, float, float, float, float, size_t, size_t,
                        size_t, unsigned int>(program, "mandelbrot");
    auto out = cl::Buffer(context, std::begin(buf), std::end(buf), true);
#endif  // use_opencl

    auto timer = std::clock();

    std::cout << '\n';

    const auto nblocks = static_cast<size_t>(std::ceil(size / bsize) + 1u);
    const auto ndig = ndigits(nblocks);

    for (size_t i = 0; i < size; i += bsize) {
      // print
      const auto blocknum = i / bsize + 1;
      std::cout << "\r[" << std::setw(3)
                << static_cast<int>((blocknum / static_cast<double>(nblocks)) *
                                    100)
                << "%] block " << std::setw(ndig) << blocknum << " of "
                << nblocks << ", " << std::min(bsize, size - i) << " pixels"
                << std::flush;

#ifdef use_opencl
      // OpenCL:
      kern(cl::EnqueueArgs(queue, cl::NDRange(bsize)), out, start_x, start_y,
           dx, dy, width, i, size, nmax);

      queue.finish();
      cl::copy(queue, out, std::begin(buf), std::end(buf));
#else   // ndef use_opencl
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
            buf[j * 3] = static_cast<unsigned char>(scaled * scaled * 255u);
            buf[j * 3 + 1] = static_cast<unsigned char>(scaled * 128u);
            buf[j * 3 + 2] = static_cast<unsigned char>(scaled * 255u);
          } else {
            buf[j * 3] = buf[j * 3 + 1] = buf[j * 3 + 2] = 0u;
          }
        }
      }
#endif  // use_opencl

      // Write block to file
      const auto io_begin = std::clock();
      unsigned char* it = buf;

      while (it != buf + std::min(bsize, size - i) * 3) {
#ifdef use_mmap
        *f++ = static_cast<char>(*it++);
#else
        file << *it++;
#endif
      }

      io_time +=
          (std::clock() - io_begin) / static_cast<double>(CLOCKS_PER_SEC);
    }

    auto duration =
        (std::clock() - timer) / static_cast<double>(CLOCKS_PER_SEC);
    auto px_per_sec = size / (duration - io_time);

    std::cout << "\r[100%] processed " << std::ceil(size / bsize) + 1
              << " blocks in " << duration << 's' << std::endl
              << "render rate = " << px_per_sec / 1e6 << " million pixels / s, "
              << int(std::floor(1 / duration)) << " fps" << std::endl;

#ifdef use_mmap
    // write output to disk
    if (msync(map, filesize, MS_SYNC) == -1)
      throw std::runtime_error{"couldn't sync to disk"};
    // free and unmap mmap & file
    if (munmap(map, filesize) == -1)
      throw std::runtime_error{"couldn't unmap memory"};
#else
    file.close();
#endif  // use_mmap
  } catch (std::exception& err) {
    std::cerr << "fatal: " << err.what() << std::endl
              << std::endl
              << "aborting." << std::endl;
    ret = 1;
  }

  if (fd != -1) close(fd);

  return ret;
}
