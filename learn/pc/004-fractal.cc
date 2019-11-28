#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <vector>

#include <phd/test>

class Pixel {
 public:
  using value_type = unsigned char;

  Pixel() : r(0), g(0), b(0) {}
  Pixel(const value_type _r, const value_type _g, const value_type _b)
      : r(_r), g(_g), b(_b) {}

  value_type r, g, b;
};

std::ostream& operator<<(std::ostream& out, const Pixel& pixel) {
  out << static_cast<int>(pixel.r) << ' ' << static_cast<int>(pixel.g) << ' '
      << static_cast<int>(pixel.b) << ' ';
  return out;
}

// A rendered image.
template <size_t _width, size_t _height>
class Image {
 public:
  const size_t width;
  const size_t height;
  const size_t size;

  Image() : width(_width), height(_height), size(_width * _height) {
    data.fill(Pixel{});
  }

  auto operator[](const size_t y) { return __arr(data, y); }

  class __arr {
   public:
    __arr(std::array<Pixel, _width * _height>& d, const size_t y)
        : _d(d), _y(y) {}

    Pixel& operator[](const size_t x) { return _d[_y * _width + x]; }

   private:
    std::array<Pixel, _width * _height>& _d;
    const size_t _y;
  };

  friend auto& operator<<(std::ostream& out,
                          const Image<_width, _height>& image) {
    // Print PPM header.

    // Magic number:
    out << "P3" << std::endl;
    // Image dimensions:
    out << image.width << " " << image.height << std::endl;
    // Max colour value:
    out << unsigned(std::numeric_limits<Pixel::value_type>::max()) << std::endl;

    // Iterate over each point in the image, writing pixel data.
    for (size_t i = 0; i < image.size; i++) {
      const Pixel pixel = image.data[i];
      out << pixel << ' ';

      // Add newline at the end of each row:
      if (!(i % image.width)) out << std::endl;
    }

    return out;
  }

 private:
  std::array<Pixel, _width * _height> data;
};

TEST(Fractals, imgTest) {
  static const size_t width = 200;
  static const size_t height = 100;

  Image<width, height> img;

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      img[y][x] = Pixel{static_cast<Pixel::value_type>(
                            (x / static_cast<double>(width)) * 255),
                        0, static_cast<Pixel::value_type>(
                               (y / static_cast<double>(height)) * 255)};
    }
  }

  std::ofstream file;
  file.open("004-imgtest.ppm");
  file << img;
  file.close();
}

TEST(Fractals, mandelbrot) {
  static const int nmax = 32;
  static const size_t width = 500, height = 500;
  static const double start_y = -1.5, end_y = 1.5;
  static const double start_x = -2, end_x = 1;
  static const double dy = (end_y - start_y) / static_cast<double>(height);
  static const double dx = (end_x - start_x) / static_cast<double>(width);
  Image<width, height> img;

  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++) {
      int n = 0;
      double r = 0, s = 0, next = 0;
      double x = start_x + i * dx, y = start_y + j * dy;

      while (r * r + s * s <= 4 && n < nmax) {
        next = r * r - s * s + x;
        s = 2 * r * s + y;
        r = next;
        ++n;
      }

      // Convert z to colour:
      double scaled = n / static_cast<double>(nmax);
      img[j][i] = Pixel{
          static_cast<Pixel::value_type>(scaled * scaled * 255u),
          static_cast<Pixel::value_type>(std::max(.25 - scaled, 0.0) * 255u),
          static_cast<Pixel::value_type>(std::min(scaled * 1.25, 1.0) * 255u)};
    }
  }

  std::ofstream file;
  file.open("004-mandelbrot.ppm");
  file << img;
  file.close();
}

PHD_MAIN();
