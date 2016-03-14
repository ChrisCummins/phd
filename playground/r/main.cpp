#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T>
struct vec3 {
  T x, y, z;

  T& operator[](const size_t i) {
    switch (i) {
      case 1: return y;
      case 2: return z;
      default: return x;
    }
  }

  vec3() : x(0), y(0), z(0) {}
  vec3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

  inline vec3<T> operator ^(const vec3<T> &rhs) const {
    return vec3<T>(y * rhs.z - z * rhs.y,
                   z * rhs.x - x * rhs.z,
                   x * rhs.y - y * rhs.x);
  }

  inline vec3<T> operator +(const vec3<T> &rhs) const {
    return vec3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
  }

  inline vec3<T> operator -(const vec3<T> &rhs) const {
    return vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  inline vec3<T> operator *(float f) const {
    return vec3<T>(x * f, y * f, z * f);
  }

  inline T operator *(const vec3<T> &rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  float norm() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  vec3<T>& normalize(T l = 1) {
    *this = (*this)*(l / norm()); return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, vec3<T>& v) {
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
    return out;
  }
};

using vec3f = vec3<float>;

class Model {
 public:
  using vertex_type = vec3f;
  using vertices_type = std::vector<vertex_type>;
  using face_type = std::vector<int>;
  using faces_type = std::vector<face_type>;

 private:
  vertices_type _verts;
  faces_type _faces;

 public:
  explicit Model(const char *filename) : _verts(), _faces() {
    std::ifstream in{filename, std::ifstream::in};
    if (in.fail())
      throw std::runtime_error{"loading model"};

    std::string line;
    while (!in.eof()) {
      std::getline(in, line);
      std::istringstream iss(line.c_str());
      char trash;
      if (!line.compare(0, 2, "v ")) {
        iss >> trash;

        vec3f v;
        for (size_t i = 0; i < 3; i++)
          iss >> v[i];

        _verts.push_back(v);
      } else if (!line.compare(0, 2, "f ")) {
        std::vector<int> f;
        int itrash, idx;
        iss >> trash;
        while (iss >> idx >> trash >> itrash >> trash >> itrash) {
          idx--;  // in wavefront obj all indices start at 1, not zero
          f.push_back(idx);
        }
        _faces.push_back(f);
      }
    }
  }

  auto verts() { return _verts; }
  auto verts() const { return _verts; }

  auto faces() { return _faces; }
  auto faces() const { return _faces; }
};


class pixel {
 public:
  using value_type = unsigned char;

  pixel() : r(0), g(0), b(0) {}
  pixel(const value_type _r, const value_type _g, const value_type _b)
      : r(_r), g(_g), b(_b) {}

  value_type r, g, b;
};

template<typename T>
class subscriptable_t {
 public:
  using pointer = T*;
  using reference = T&;

  explicit subscriptable_t(pointer root) : _root(root) {}

  reference operator[](const size_t x) { return _root[x]; }

 private:
  pointer _root;
};

std::ostream& operator<<(std::ostream& out, const pixel& pixel) {
  out << static_cast<int>(pixel.r)
      << ' ' << static_cast<int>(pixel.g)
      << ' ' << static_cast<int>(pixel.b) << ' ';
  return out;
}


class Image {
 public:
  Image(const size_t width, const size_t height,
        const pixel& fill = pixel{})
      : _width(width), _height(height),
        _data(width * height, fill) {}

  size_t width() const { return _width; }
  size_t height() const { return _height; }
  size_t size() const { return width() * height(); }

  auto operator[](const size_t y) {
    return subscriptable_t<pixel>(_data.data() + (height() - y) * width());
  }

  friend auto& operator<<(std::ostream& out, const Image& img) {
    // PPM header.

    out << "P3" << std::endl << img.width() << ' ' << img.height() << std::endl
        << unsigned(std::numeric_limits<pixel::value_type>::max()) << std::endl;

    // Iterate over each point in the img, writing pixel data.
    for (size_t i = 0; i < img.size(); i++) {
      const pixel pixel = img._data[i];
      out << pixel << ' ';

      // Add newline at the end of each row:
      if (!(i % img.width()))
        out << std::endl;
    }

    return out;
  }

 private:
  const size_t _width, _height;
  std::vector<pixel> _data;
};

void line(Image& img, int x0, int y0, int x1, int y1, pixel color) {
  bool steep = false;
  if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
    std::swap(x0, y0);
    std::swap(x1, y1);
    steep = true;
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }
  int dx = x1 - x0;
  int dy = y1 - y0;
  int derror2 = std::abs(dy) * 2;
  int error2 = 0;
  int y = y0;
  for (int x = x0; x <= x1; ++x) {
    if (steep) {
      img[size_t(x)][size_t(y)] = color;
    } else {
      img[size_t(y)][size_t(x)] = color;
    }

    error2 += derror2;
    if (error2 > dx) {
      y += (y1 >y0 ? 1 : -1);
      error2 -= dx*2;
    }
  }
}

int main() {
  auto timer = std::clock();

  static const size_t width = 2048, height = 2048;
  Image img{width, height};
  Model model{"african_head.obj"};

  for (auto& face : model.faces()) {
    for (size_t j = 0; j < 3; ++j) {
      vec3f v0 = model.verts()[size_t(face[size_t(j)])];
      vec3f v1 = model.verts()[size_t(face[(j + 1) % 3])];
      int x0 = static_cast<int>((v0.x + 1.) * width / 2.);
      int y0 = static_cast<int>((v0.y + 1.) * height / 2.);
      int x1 = static_cast<int>((v1.x + 1.) * width / 2.);
      int y1 = static_cast<int>((v1.y + 1.) * height / 2.);

      line(img, x0, y0, x1, y1, pixel{255, 255, 255});
    }
  }

  std::ofstream file{"render.ppm"};
  file << img;
  file.close();

  auto duration = (std::clock() - timer)
                  / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "completed in " << duration << "s" << std::endl;

  return 0;
}
