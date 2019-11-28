/* -*-c++-*-
 *
 * Copyright (C) 2015, 2016 Chris Cummins.
 *
 * This file is part of rt.
 *
 * rt is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * rt is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rt.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef OBJECTS_H_
#define OBJECTS_H_

#include <vector>

#include "rt/graphics.h"
#include "rt/math.h"
#include "rt/profiling.h"
#include "rt/ray.h"
#include "rt/restrict.h"

namespace rt {

// Properties that describe a material.
class Material {
 public:
  const Colour colour;
  const Scalar ambient;       // 0 <= ambient <= 1
  const Scalar diffuse;       // 0 <= diffuse <= 1
  const Scalar specular;      // 0 <= specular <= 1
  const Scalar shininess;     // shininess >= 0
  const Scalar reflectivity;  // 0 <= reflectivity < 1

  // Constructor.
  inline Material(const Colour &_colour, const Scalar _ambient,
                  const Scalar _diffuse, const Scalar _specular,
                  const Scalar _shininess, const Scalar _reflectivity)
      : colour(_colour),
        ambient(_ambient),
        diffuse(_diffuse),
        specular(_specular),
        shininess(_shininess),
        reflectivity(_reflectivity) {}
};

// A physical object that light interacts with.
class Object {
 public:
  const Vector position;

  // Constructor.
  explicit inline Object(const Vector &_position) : position(_position) {
    // Register object with profiling counter.
    profiling::counters::incObjectsCount();
  }

  // Virtual destructor.
  virtual ~Object() {}

  // Return surface normal at point p.
  virtual Vector normal(const Vector &p) const = 0;
  // Return whether ray intersects object, and if so, at what
  // distance (0 if no intersect).
  virtual Scalar intersect(const Ray &ray) const = 0;
  // Return material at point on surface.
  virtual const Material *surface(const Vector &point) const = 0;
};

using Objects = const std::vector<Object *>;

// A plane.
class Plane : public Object {
 public:
  const Vector direction;
  const Material *const restrict material;

  // Constructor.
  inline Plane(const Vector &_origin, const Vector &_direction,
               const Material *const restrict _material)
      : Object(_origin),
        direction(_direction.normalise()),
        material(_material) {}

  virtual inline Vector normal(const Vector &p) const { return direction; }

  virtual inline Scalar intersect(const Ray &ray) const {
    // Calculate intersection of line and plane.
    const Scalar f = (position - ray.position) ^ direction;
    const Scalar g = ray.direction ^ direction;
    const Scalar t = f / g;

    // Accommodate for precision errors.
    const Scalar t0 = t - ScalarPrecision / 2;
    const Scalar t1 = t + ScalarPrecision / 2;

    if (t0 > ScalarPrecision)
      return t0;
    else if (t1 > ScalarPrecision)
      return t1;
    else
      return 0;
  }

  virtual inline const Material *surface(const Vector &point) const {
    return material;
  }
};

class CheckerBoard : public Plane {
 public:
  const Material *const restrict material1;
  const Material *const restrict material2;
  const Scalar checkerWidth;

  inline CheckerBoard(const Vector &_origin, const Vector &_direction,
                      const Scalar _checkerWidth,
                      const Material *const restrict _material1,
                      const Material *const restrict _material2)
      : Plane(_origin, _direction, nullptr),
        material1(_material1),
        material2(_material2),
        checkerWidth(_checkerWidth) {}

  inline ~CheckerBoard() {}

  virtual inline const Material *surface(const Vector &point) const {
    // TODO: translate point to a relative position on plane.
    const Vector relative =
        Vector(point.x + gridOffset, point.z + gridOffset, 0);

    const int x = static_cast<int>(relative.x);
    const int y = static_cast<int>(relative.y);
    const int half = static_cast<int>(checkerWidth * 2);
    const int mod = half * 2;

    if (x % mod < half)
      return y % mod < half ? material1 : material2;
    else
      return y % mod < half ? material2 : material1;
  }

 private:
  static const Scalar gridOffset;
};

// A sphere consits of a position and a radius.
class Sphere : public Object {
 public:
  const Scalar radius;
  const Material *const restrict material;

  // Constructor.
  inline Sphere(const Vector &_position, const Scalar _radius,
                const Material *const restrict _material)
      : Object(_position), radius(_radius), material(_material) {}

  virtual inline Vector normal(const Vector &p) const {
    return (p - position).normalise();
  }

  virtual inline Scalar intersect(const Ray &ray) const {
    // Calculate intersection of line and sphere.
    const Vector distance = position - ray.position;
    const Scalar b = ray.direction ^ distance;
    const Scalar d = b * b + radius * radius - (distance ^ distance);

    if (d < 0) return 0;

    const Scalar t0 = b - sqrt(d);
    const Scalar t1 = b + sqrt(d);

    if (t0 > ScalarPrecision)
      return t0;
    else if (t1 > ScalarPrecision)
      return t1;
    else
      return 0;
  }

  virtual inline const Material *surface(const Vector &point) const {
    return material;
  }
};

}  // namespace rt

#endif  // OBJECTS_H_
