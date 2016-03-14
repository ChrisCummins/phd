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
class vec2 {
 public:
  using value_type = T;

  value_type x, y;

  value_type& operator[](const size_t i) {
    switch (i) {
      case 1: return y;
      default: return x;
    }
  }

  explicit vec2(const value_type& fill = value_type{}) : x(fill), y(fill) {}

  vec2(const value_type& _x, const value_type& _y) : x(_x), y(_y) {}

  inline vec2 operator+(const vec2& rhs) const {
    return vec2(x + rhs.x, y + rhs.y);
  }

  inline vec2 operator-(const vec2& rhs) const {
    return vec2(x - rhs.x, y - rhs.y);
  }

  inline vec2 operator*(const float f) const {
    return vec2(x * f, y * f);
  }

  inline value_type operator*(const vec2& rhs) const {
    return x * rhs.x + y * rhs.y;
  }

  float norm() const {
    return std::sqrt(x * x + y * y);
  }

  vec2& normalize(const value_type& l = value_type{1}) {
    *this = *this * (l / norm());
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const vec2& v) {
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
    return out;
  }
};

using vec2f = vec2<float>;


template<typename T>
class vec3 {
 public:
  using value_type = T;

  value_type x, y, z;

  value_type& operator[](const size_t i) {
    switch (i) {
      case 1: return y;
      case 2: return z;
      default: return x;
    }
  }

  explicit vec3(const value_type& fill = value_type{})
      : x(fill), y(fill), z(fill) {}

  vec3(const value_type& _x, const value_type& _y, const value_type& _z)
      : x(_x), y(_y), z(_z) {}

  inline vec3 operator^(const vec3& rhs) const {
    return vec3(y * rhs.z - z * rhs.y,
                z * rhs.x - x * rhs.z,
                x * rhs.y - y * rhs.x);
  }

  inline vec3 operator+(const vec3& rhs) const {
    return vec3(x + rhs.x, y + rhs.y, z + rhs.z);
  }

  inline vec3 operator-(const vec3& rhs) const {
    return vec3(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  inline vec3 operator*(const float f) const {
    return vec3(x * f, y * f, z * f);
  }

  inline value_type operator*(const vec3& rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  float norm() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  vec3& normalize(const value_type& l = value_type{1}) {
    *this = *this * (l / norm());
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const vec3& v) {
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
    return out;
  }
};

using vec3f = vec3<float>;


class Model {
 public:
  using vertices_type = std::vector<vec3f>;
  using faces_type = std::vector<std::vector<int>>;

  explicit Model(const std::string& filename) {
    std::ifstream in{filename, std::ifstream::in};
    if (in.fail())
      throw std::runtime_error{"loading model"};

    std::string line;

    while (!in.eof()) {
      std::getline(in, line);
      std::istringstream iss(line.c_str());

      char trash;  // unused
      if (!line.compare(0, 2, "v ")) {
        iss >> trash;

        // parse vector
        vec3f v;
        for (size_t i = 0; i < 3; i++)
          iss >> v[i];

        _verts.push_back(v);
      } else if (!line.compare(0, 2, "f ")) {
        std::vector<int> f;
        int itrash, idx;
        iss >> trash;

        // parse face
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

 private:
  vertices_type _verts;
  faces_type _faces;
};


class pixel {
 public:
  using value_type = unsigned char;
  value_type r, g, b;

  explicit pixel(const value_type fill = 0) : r(fill), g(fill), b(fill) {}
  pixel(const value_type _r, const value_type _g, const value_type _b)
      : r(_r), g(_g), b(_b) {}

  // human-readable:
  friend std::ostream& operator<<(std::ostream& out, const pixel& pixel) {
    out << static_cast<int>(pixel.r)
        << ' ' << static_cast<int>(pixel.g)
        << ' ' << static_cast<int>(pixel.b) << ' ';
    return out;
  }
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


class Image {
 public:
  Image(const size_t width, const size_t height,
        bool inverted = true, const pixel& fill = pixel{})
      : _width(width), _height(height), _inverted(inverted),
        _data(width * height, fill) {}

  size_t width() const { return _width; }
  size_t height() const { return _height; }
  size_t size() const { return width() * height(); }

  auto operator[](const size_t y) {
    if (_inverted)
      return subscriptable_t<pixel>{_data.data() + (height() - y) * width()};
    else
      return subscriptable_t<pixel>{_data.data() + y * width()};
  }

  // P6 file format:
  friend auto& operator<<(std::ostream& out, const Image& img) {
    out << "P6\n" << img.width() << ' ' << img.height() << '\n'
        << int(std::numeric_limits<pixel::value_type>::max()) << '\n';

    for (const auto& pixel : img._data)
      out << pixel.r << pixel.g << pixel.b;

    return out;
  }

 private:
  const size_t _width, _height;
  const bool _inverted;
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
    if (steep)
      img[size_t(x)][size_t(y)] = color;
    else
      img[size_t(y)][size_t(x)] = color;

    error2 += derror2;
    if (error2 > dx) {
      y += (y1 > y0 ? 1 : -1);
      error2 -= dx * 2;
    }
  }
}


void wireframe(Image& img, Model& model) {
  for (auto& face : model.faces()) {
    for (size_t j = 0; j < 3; ++j) {
      vec3f v0 = model.verts()[size_t(face[size_t(j)])];
      vec3f v1 = model.verts()[size_t(face[(j + 1) % 3])];

      int x0 = static_cast<int>((v0.x + 1.) * img.width() / 2);
      int y0 = static_cast<int>((v0.y + 1.) * img.height() / 2);
      int x1 = static_cast<int>((v1.x + 1.) * img.width() / 2);
      int y1 = static_cast<int>((v1.y + 1.) * img.height() / 2);

      line(img, x0, y0, x1, y1, pixel{255, 255, 255});
    }
  }
}

int main() {
  static const size_t width = 2048, height = 2048;

  auto start = std::clock(); double timer;

  Image img{width, height};
  Model model{"african_head.obj"};

  wireframe(img, model);

  std::ofstream file{"render.ppm"};
  file << img;
  file.close();

  timer = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "completed in " << timer << "s" << std::endl;

  return 0;
}
