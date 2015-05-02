// -*- c-basic-offset: 8; -*-
#ifndef OBJECTS_H_
#define OBJECTS_H_

#include "./math.h"
#include "./ray.h"
#include "./graphics.h"

// Profiling counter.
extern uint64_t objectsCount;

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
        inline Material(const Colour &colour,
                 const Scalar ambient,
                 const Scalar diffuse,
                 const Scalar specular,
                 const Scalar shininess,
                 const Scalar reflectivity)
                : colour(colour),
                  ambient(ambient),
                  diffuse(diffuse),
                  specular(specular),
                  shininess(shininess),
                  reflectivity(reflectivity) {}
};

// A physical object that light interacts with.
class Object {
 public:
        const Vector position;

        // Constructor.
        explicit inline Object(const Vector &position)
                : position(position) {
            // Register object with profiling counter.
            objectsCount += 1;
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

// A plane.
class Plane : public Object {
 public:
        const Vector direction;
        const Material *const material;

        // Constructor.
        inline Plane(const Vector &origin,
                     const Vector &direction,
                     const Material *const material)
                : Object(origin),
                  direction(direction.normalise()),
                  material(material) {}

        virtual inline Vector normal(const Vector &p) const {
            return direction;
        }

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
        const Material *const material1;
        const Material *const material2;
        const Scalar checkerWidth;

        inline CheckerBoard(const Vector &origin,
                            const Vector &direction,
                            const Scalar checkerWidth,
                            const Material *const material1,
                            const Material *const material2)
                : Plane(origin, direction, nullptr),
                  material1(material1), material2(material2),
                  checkerWidth(checkerWidth) {}

        inline ~CheckerBoard() {}

        virtual inline const Material *surface(const Vector &point) const {
            // TODO: translate point to a relative position on plane.
            const Vector relative = Vector(point.x + gridOffset,
                                           point.z + gridOffset, 0);

            const int x = relative.x;
            const int y = relative.y;
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
        const Material *const material;

        // Constructor.
        inline Sphere(const Vector &position,
                      const Scalar radius,
                      const Material *const material)
                : Object(position), radius(radius), material(material) {}

        virtual inline Vector normal(const Vector &p) const {
            return (p - position).normalise();
        }

        virtual inline Scalar intersect(const Ray &ray) const {
            // Calculate intersection of line and sphere.
            const Vector distance = position - ray.position;
            const Scalar b = ray.direction ^ distance;
            const Scalar d = b * b + radius * radius - (distance ^ distance);

            if (d < 0)
                return 0;

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

#endif  // OBJECTS_H_
