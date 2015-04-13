// -*- c-basic-offset: 8; -*-
#ifndef _RT_H_
#define _RT_H_

#include <cstdio>
#include <random>
#include <stdint.h>
#include <vector>

// A simple ray tacer. Features:
//
//   * Objects: Spheres & Planes.
//   * Lighting: point light and soft lights.
//   * Shading: Lambert (diffuse) and Phong (specular).
//   * Anti-aliasing: Stochastic supersampling.


// BASIC TYPES.

// Individual real numbers are known as scalars. Precision can be
// controlled by changing the data type between different floating
// point types.
typedef double Scalar;

// The "rounding error" to accomodate for when approximate infinite
// precision real numbers.
static const Scalar ScalarPrecision = 1e-6;

// A vector consits of three coordinate scalars. Vectors are immutable.
class Vector {
public:
        const Scalar x;
        const Scalar y;
        const Scalar z;

        // Contructor: V = (x,y,z)
        Vector(const Scalar x, const Scalar y, const Scalar z);

        // Addition: A' = A + B
        Vector inline operator+(const Vector &b) const;

        // Subtraction: A' = A - B
        Vector inline operator-(const Vector &b) const;

        // Multiplication: A' = aA
        Vector inline operator*(const Scalar a) const;

        // Division: A' = A / a
        Vector inline operator/(const Scalar a) const;

        // Product: A' = (Ax * Bx, Ay * By, Az * Bz)
        Vector inline operator*(const Vector &b) const;

        // Dot product: x = A . B
        Scalar inline operator^(const Vector &b) const;

        // Cross product: A' = A x B
        Vector inline operator|(const Vector &b) const;

        // Equality: A == B
        bool inline operator==(const Vector &b) const;

        // Inequality: A != B
        bool inline operator!=(const Vector &b) const;

        // Length of vector: |A| = sqrt(x^2 + y^2 + z^2)
        Scalar inline size() const;

        // Product of components: x * y * z
        Scalar inline product() const;

        // Sum of components: x + y + z
        Scalar inline sum() const;

        // Normalise: A' = A / |A|
        Vector inline normalise() const;
};

// RANDOM NUMBERS.

// The distribution type for generating random numbers.
typedef std::normal_distribution<Scalar> DistributionType;

// A random number generator for sampling a normal distribution.
class NormalDistribution {
public:
        // Construct a normal distribution about the range [min,max].
        NormalDistribution(const Scalar min, const Scalar max);

        // Return a sample from the distribution.
        Scalar operator()();
private:
        std::mt19937 generator;
        DistributionType distribution;
};

// GRAPHICS TYPES.

// The output type of a single R,G,B colour component.
typedef uint8_t PixelColourType;

// The maximum value of a single R,G,B colour component.
static const uint8_t PixelColourMax = 255;

// Clamp a Scalar value to within the range [0,1].
Scalar inline clamp(const Scalar x);

// Transform a scalar from the range [0,1] to [0,PixelColourMax]. Note
// that this transformation may be non-linear.
PixelColourType inline scale(const Scalar x);

// A pixel is a trio of R,G,B components.
struct Pixel { PixelColourType r, g, b; };

// A colour is represented by R,G,B scalars, and are mutable through
// the += and /= operators. They behave identically to Vectors.
class Colour {
public:
        Scalar r, g, b;

        // Constructor for specifying colours as 32 bit hex
        // string. E.g. 0xff00aa.
        Colour(const int hex);

        // Contructor: C = (r,g,b)
        Colour(const float r=0, const float g=0, const float b=0);

        // Colour addition.
        void operator+=(const Colour &c);

        // Scalar division.
        void operator/=(const Scalar x);

        // Scalar colour multiplication.
        Colour operator*(const Scalar x) const;

        // Scalar colour divison.
        Colour operator/(const Scalar x) const;

        // Combination of two colours: A' = (Ar * Br, Ag * Bg, Ab * Bb)
        Colour operator*(const Colour &rhs) const;

        // Explicit cast operation from Colour -> Pixel.
        explicit operator Pixel() const;
};

// Properties that describe a material.
class Material {
public:
        const Colour colour;
        const Scalar ambient;      // 0 <= ambient <= 1
        const Scalar diffuse;      // 0 <= diffuse <= 1
        const Scalar specular;     // 0 <= specular <= 1
        const Scalar shininess;    // shininess >= 0
        const Scalar reflectivity; // 0 <= reflectivity < 1

        // Constructor.
        Material(const Colour &colour,
                 const Scalar ambient,
                 const Scalar diffuse,
                 const Scalar specular,
                 const Scalar shininess,
                 const Scalar reflectivity);
};

// A ray abstraction.
class Ray {
public:
        const Vector position, direction;

        // Constructors.
        Ray(const Scalar x=0, const Scalar y=0);
        Ray(const Vector &position, const Vector &direction);
};

// A physical object that light interacts with.
class Object {
public:
        const Vector position;

        // Constructor.
        Object(const Vector &position);

        // Virtual destructor.
        virtual ~Object() {};

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
        Plane(const Vector &origin,
              const Vector &direction,
              const Material *const material);

        virtual Vector normal(const Vector &p) const;
        virtual Scalar intersect(const Ray &ray) const;
        virtual const Material *surface(const Vector &point) const;
};

class CheckerBoard : public Plane {
public:
        const Material *const black;
        const Material *const white;
        const Scalar checkerWidth;

        CheckerBoard(const Vector &origin,
                     const Vector &direction,
                     const Scalar checkerWidth);
        ~CheckerBoard();

        virtual const Material *surface(const Vector &point) const;
};

// A sphere consits of a position and a radius.
class Sphere : public Object {
public:
        const Scalar radius;
        const Material *const material;

        // Constructor.
        Sphere(const Vector &position,
               const Scalar radius,
               const Material *const material);

        virtual Vector normal(const Vector &p) const;
        virtual Scalar intersect(const Ray &ray) const;
        virtual const Material *surface(const Vector &point) const;
};

// Base class light source.
class Light {
public:
    // Virtual destructor.
    virtual ~Light() {};

    // Calculate the shading colour at `point' for a given surface
    // material, surface normal, and direction to the ray.
    virtual Colour shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const = 0;
};

// A point light source.
class PointLight : public Light {
public:
    const Vector position;
    const Colour colour;

    // Constructor.
    PointLight(const Vector &position,
               const Colour &colour=Colour(0xff, 0xff, 0xff));

    virtual Colour shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const;
};

// A round light source.
class SoftLight : public Light {
public:
        const Vector position;
        const Scalar radius;
        const Colour colour;
        const size_t samples;

        // Constructor.
        SoftLight(const Vector &position, const Scalar radius,
                  const Colour &colour=Colour(0xff, 0xff, 0xff));

        virtual Colour shade(const Vector &point,
                             const Vector &normal,
                             const Vector &toRay,
                             const Material *const material,
                             const std::vector<const Object *> objects) const;
};

// A full scene, consisting of objects (spheres) and lighting (point
// lights).
class Scene {
public:
        const std::vector<const Object *> objects;
        const std::vector<const Light *> lights;

        // Constructor.
        Scene(const std::vector<const Object *> &objects,
              const std::vector<const Light *> &lights);
};

class Renderer {
public:
        const Scene scene;

        Renderer(const Scene scene);

        // The heart of the raytracing engine.
        void render(FILE *const out) const;

private:
        const size_t width, height;

        // Trace a ray trough a given scene and return the final
        // colour.
        Colour trace(const Ray &ray,
                     Colour colour=Colour(0, 0, 0),
                     const unsigned int depth=0) const;

        // Calculate the colour at position x,y through supersampling.
        Colour supersample(size_t x, size_t y) const;
};

// Return the index of the object with the closest intersection, and
// the distance to the intersection `t'. If no intersection, return
// -1.
int closestIntersect(const Ray &ray,
                     const std::vector<const Object *> &objects,
                     Scalar &t);

// Return whether a given ray intersects any of the objects.
bool intersects(const Ray &ray, const std::vector<const Object *> &objects);

#endif  // _RT_H_
