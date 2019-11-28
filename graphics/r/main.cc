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

template <typename T>
class vec2 {
 public:
  using value_type = T;

  value_type x, y;

  value_type& operator[](const size_t i) {
    switch (i) {
      case 1:
        return y;
      default:
        return x;
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
    return vec2(static_cast<int>(x * f), static_cast<int>(y * f));
  }

  inline value_type operator*(const vec2& rhs) const {
    return x * rhs.x + y * rhs.y;
  }

  // implicit conversion between types
  template <typename U>
  operator vec2<U>() const {
    return vec2<U>{static_cast<U>(x), static_cast<U>(y)};
  }

  float norm() const { return std::sqrt(x * x + y * y); }

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
using vec2i = vec2<int>;

template <typename T>
class vec3 {
 public:
  using value_type = T;

  value_type x, y, z;

  value_type& operator[](const size_t i) {
    switch (i) {
      case 1:
        return y;
      case 2:
        return z;
      default:
        return x;
    }
  }

  explicit vec3(const value_type& fill = value_type{})
      : x(fill), y(fill), z(fill) {}

  vec3(const value_type& _x, const value_type& _y, const value_type& _z)
      : x(_x), y(_y), z(_z) {}

  // cross product
  inline vec3 operator^(const vec3& rhs) const {
    return vec3(y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z,
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

  // implicit conversion between types
  template <typename U>
  operator vec3<U>() const {
    return vec3<U>{static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
  }

  float norm() const { return std::sqrt(x * x + y * y + z * z); }

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

vec3f barycentric(vec3f a, vec3f b, vec3f c, vec3f p) {
  vec3f s[2];
  for (unsigned int i = 2; i--;) {
    s[i][0] = c[i] - a[i];
    s[i][1] = b[i] - a[i];
    s[i][2] = a[i] - p[i];
  }
  vec3f u = s[0] ^ s[1];

  // dont forget that u[2] is integer. If it is zero then triangle ABC
  // is degenerate:
  if (std::abs(u[2]) > 1e-2)
    return vec3f(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
  // in this case generate negative coordinates, it will be thrown
  // away by the rasterizer:
  return vec3f(-1, 1, 1);
}

vec3f barycentric(vec2f a, vec2f b, vec2f c, vec2f p) {
  vec3f u = vec3f{c[0] - a[0], b[0] - a[0], a[0] - p[0]}  // NOLINT
            ^ vec3f { c[1] - a[1], b[1] - a[1], a[1] - p[1] };
  // triangle is degenerate, in this case return smth with negative
  // coordinates:
  if (std::abs(u[2]) < 1) return vec3f(-1, 1, 1);
  return vec3f(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

class Model {
 public:
  using vertices_type = std::vector<vec3f>;
  using faces_type = std::vector<std::vector<int>>;

  explicit Model(const std::string& filename) {
    std::ifstream in{filename, std::ifstream::in};
    if (in.fail()) throw std::runtime_error{"loading model"};

    std::string line;

    while (!in.eof()) {
      std::getline(in, line);
      std::istringstream iss(line.c_str());

      char trash;  // unused
      if (!line.compare(0, 2, "v ")) {
        iss >> trash;

        // parse vector
        vec3f v;
        for (size_t i = 0; i < 3; i++) iss >> v[i];

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

  inline pixel operator*(const float f) const {
    return pixel{static_cast<value_type>(r * f), static_cast<value_type>(g * f),
                 static_cast<value_type>(b * f)};
  }

  // human-readable:
  friend std::ostream& operator<<(std::ostream& out, const pixel& pixel) {
    out << static_cast<int>(pixel.r) << ' ' << static_cast<int>(pixel.g) << ' '
        << static_cast<int>(pixel.b) << ' ';
    return out;
  }
};

namespace colors {
static const pixel black;
static const pixel white{255};
static const pixel red{255, 0, 0};
static const pixel blue{0, 255, 0};
static const pixel green{0, 0, 255};
}  // namespace colors

template <typename T>
class subscriptable_t {
 public:
  using value_type = T;
  using pointer = value_type*;
  using reference = value_type&;

  explicit subscriptable_t(pointer root) : _root(root) {}
  reference operator[](const size_t x) { return _root[x]; }

 private:
  pointer _root;
};

template <typename T>
class col_iterator {
 public:
  using value_type = T;

  explicit col_iterator(value_type* data) : _data(data) {}

  col_iterator& operator++() {
    ++_data;
    return *this;
  }

  col_iterator operator++(int) {
    auto tmp = col_iterator{_data};
    operator++();
    return tmp;
  }

  value_type& operator*() { return *_data; }

  friend bool operator==(const col_iterator& lhs, const col_iterator& rhs) {
    return lhs._data == rhs._data;
  }

  friend bool operator!=(const col_iterator& lhs, const col_iterator& rhs) {
    return !(lhs == rhs);
  }

 private:
  pixel* _data;
};

template <typename T>
class row_iterator {
 public:
  using value_type = T;

  row_iterator(value_type* data, size_t stride)
      : _data(data), _stride(stride) {}
  explicit row_iterator(value_type* data) : _data(data), _stride(0) {}

  row_iterator& operator++() {
    _data += _stride;
    return *this;
  }

  row_iterator operator++(int) {
    auto tmp = row_iterator(_data, _stride);
    operator++();
    return tmp;
  }

  col_iterator<value_type> begin() { return col_iterator<value_type>{_data}; }

  col_iterator<value_type> end() {
    return col_iterator<value_type>{_data + _stride};
  }

  col_iterator<value_type> operator*() {
    return col_iterator<value_type>{_data};
  }

  friend bool operator==(const row_iterator& lhs, const row_iterator& rhs) {
    return lhs._data == rhs._data;
  }

  friend bool operator!=(const row_iterator& lhs, const row_iterator& rhs) {
    return !(lhs == rhs);
  }

 private:
  pixel* _data;
  size_t _stride;
};

class Canvas {
 public:
  using iterator = row_iterator<pixel>;

  Canvas(const size_t width, const size_t height, bool inverted = true,
         const pixel& fill = pixel{})
      : _width(width),
        _height(height),
        _inverted(inverted),
        _data(width * height, fill),
        _zbuffer(width * height, -std::numeric_limits<float>::max()) {}

  Canvas(const size_t width, const size_t height, const pixel& fill,
         bool inverted = true)
      : _width(width),
        _height(height),
        _inverted(inverted),
        _data(width * height, fill),
        _zbuffer(width * height, -std::numeric_limits<float>::max()) {}

  size_t width() const { return _width; }
  size_t height() const { return _height; }
  size_t size() const { return width() * height(); }

  // Pixel accessor
  subscriptable_t<pixel> operator[](const size_t y) {
    if (_inverted)
      return subscriptable_t<pixel>{_data.data() + (height() - y) * width()};
    else
      return subscriptable_t<pixel>{_data.data() + y * width()};
  }

  iterator begin() { return iterator{_data.data(), width()}; }

  iterator end() { return iterator{_data.data() + _data.size()}; }

  void line(int x0, int y0, int x1, int y1, const pixel& color) {
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
        (*this)[size_t(x)][size_t(y)] = color;
      else
        (*this)[size_t(y)][size_t(x)] = color;

      error2 += derror2;
      if (error2 > dx) {
        y += (y1 > y0 ? 1 : -1);
        error2 -= dx * 2;
      }
    }
  }

  void triangle(vec3f t0, vec3f t1, vec3f t2, const pixel& color) {
    vec2f bboxmin(std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max());
    vec2f bboxmax(-std::numeric_limits<float>::max(),
                  -std::numeric_limits<float>::max());
    vec2f clamp(static_cast<float>(width() - 1),
                static_cast<float>(height() - 1));

    for (unsigned int j = 0; j < 2; ++j) {
      bboxmin[j] = std::max(0.0f, std::min(bboxmin[j], t0[j]));
      bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], t0[j]));
    }
    for (unsigned int j = 0; j < 2; ++j) {
      bboxmin[j] = std::max(0.0f, std::min(bboxmin[j], t1[j]));
      bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], t1[j]));
    }
    for (unsigned int j = 0; j < 2; ++j) {
      bboxmin[j] = std::max(0.0f, std::min(bboxmin[j], t2[j]));
      bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], t2[j]));
    }

    vec3f p;
    for (p.x = bboxmin.x; p.x <= bboxmax.x; ++p.x) {
      for (p.y = bboxmin.y; p.y <= bboxmax.y; ++p.y) {
        vec3f bc_screen = barycentric(t0, t1, t2, p);
        if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

        p.z = t0[2] * bc_screen[0];
        p.z += t1[2] * bc_screen[1];
        p.z += t2[2] * bc_screen[2];

        if (_zbuffer[size_t(p.x + p.y * width())] < p.z) {
          _zbuffer[size_t(p.x + p.y * width())] = static_cast<int>(p.z);
          (*this)[size_t(p.y)][size_t(p.x)] = color;
        }
      }
    }
  }

  void wireframe(const Model& model) {
    for (const auto& face : model.faces()) {
      for (size_t j = 0; j < 3; ++j) {
        vec3f v0 = model.verts()[size_t(face[size_t(j)])];
        vec3f v1 = model.verts()[size_t(face[(j + 1) % 3])];

        const auto x0 = static_cast<int>((v0.x + 1) * width() / 2);
        const auto y0 = static_cast<int>((v0.y + 1) * height() / 2);
        const auto x1 = static_cast<int>((v1.x + 1) * width() / 2);
        const auto y1 = static_cast<int>((v1.y + 1) * height() / 2);

        line(x0, y0, x1, y1, pixel{255, 255, 255});
      }
    }
  }

  vec3f world2screen(vec3f v) {
    return vec3f{static_cast<int>((v.x + 1) * width() / 2 + 0.5),
                 static_cast<int>((v.y + 1.) * height() / 2 + 0.5), v.z};
  }

  void solid(const Model& model, const vec3f& light_direction,
             const pixel& surface_color) {
    for (const auto& face : model.faces()) {
      vec3f screen_coords[3];
      vec3f world_coords[3];

      for (unsigned int i = 0; i < 3; ++i) {
        vec3f v = model.verts()[size_t(face[i])];
        world_coords[i] = v;
        screen_coords[i] = world2screen(v);
      }

      // normal of face:
      vec3f n = (world_coords[2] - world_coords[0]) ^
                (world_coords[1] - world_coords[0]);
      n.normalize();

      float intensity = n * light_direction;

      // back-face culling
      if (intensity > 0) {
        triangle(screen_coords[0], screen_coords[1], screen_coords[2],
                 surface_color * intensity);
      }
    }
  }

 protected:
  const size_t _width, _height;
  const bool _inverted;
  std::vector<pixel> _data;
  std::vector<float> _zbuffer;
};

class Image : public Canvas {
 public:
  Image(const size_t width, const size_t height, bool inverted = true,
        const pixel& fill = pixel{})
      : Canvas(width, height, inverted, fill) {}
  Image(const size_t width, const size_t height, const pixel& fill,
        bool inverted = true)
      : Canvas(width, height, inverted, fill) {}

  // P6 file format:
  friend auto& operator<<(std::ostream& out, const Image& img) {
    out << "P6\n"
        << img.width() << ' ' << img.height() << '\n'
        << int(std::numeric_limits<pixel::value_type>::max()) << '\n';

    for (const auto& pixel : img._data) out << pixel.r << pixel.g << pixel.b;

    return out;
  }
};

int main() {
  static const size_t width = 2048, height = 2048;

  auto start = std::clock();
  double timer;

  Image img{width, height, pixel{255}};
  Model model{"playground/r/african_head.obj"};

  img.solid(model, {0, 0, -1}, {255, 180, 140});

  std::ofstream file{"render.ppm"};
  file << img;
  file.close();

  timer = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "completed in " << timer << "s" << std::endl;

  return 0;
}
