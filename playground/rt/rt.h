#ifndef _RT_H_
#define _RT_H_

#include <algorithm>
#include <math.h>
#include <cstdio>
#include <stdint.h>
#include <cassert>
#include <vector>

// A pixel is a trio of R,G,B bytes.
struct Pixel { uint8_t r, g, b; };

// Colour class. Note that colours are NOT immutable.
class Colour {
public:
        float r, g, b;

        // Colour constructor.
        Colour(const float r=0, const float g=0, const float b=0);

        // Colour addition.
        void operator+=(const Colour &c);

        // Scalar colour multiplication.
        Colour operator*(const double x) const;

        // Combination of two colours.
        Colour operator*(const Colour c) const;

        // Explicit cast operation from Colour -> Pixel.
        explicit operator Pixel() const;
};

// Properties that describe a material.
class Material {
public:
        const Colour diffuse;
        const double diffuseCoefficient;
        const double specularCoefficient;

        // Constructor.
        Material(const Colour &diffuse, const double diffuseCoefficient,
                 const double specularCoefficient);
};

// Vector class.
class Vector {
public:
        const double x;
        const double y;
        const double z;

        // Constructor.
        Vector(const double x, const double y, const double z);

        //Vector& operator=(const Vector& rhs);

        // Vector addition.
        Vector operator+(const Vector &b) const;

        // Vector subtraction.
        Vector operator-(const Vector &b) const;

        // Scalar multiplication.
        Vector operator*(const double a) const;

        // Scalar division.
        Vector operator/(const double a) const;

        // Vector product.
        Vector operator*(const Vector &b) const;

        // Dot product.
        double operator^(const Vector &b) const;

        // Cross product.
        Vector operator|(const Vector &b) const;

        // Equality testing.
        bool operator==(const Vector &b) const;

        // Negative equality testing.
        bool operator!=(const Vector &b) const;

        // Length of vector.
        double magnitude() const;

        // Scalar product of components.
        double product() const;

        // Scalar sum of components.
        double sum() const;

        // Normalise vector.
        Vector normalise() const;
};


// A ray abstraction.
class Ray {
public:
        Vector position, direction;

        Ray(const double x=0, const double y=0);
        Ray(const Vector &position, const Vector &direction);
};

class Object {
public:
        const Vector position;
        const Material material;

        Object(const Vector &position, const Material &material);

        // Return surface normal at point p.
        virtual Vector normal(const Vector &p) const;
        // Return whether ray intersects object, and if so, at what
        // distance (0 if no intersect).
        virtual double intersect(const Ray &ray) const;
};

// A sphere consits of a position and a radius.
class Sphere : public Object {
public:
        const double radius;

        // Constructor.
        Sphere(const Vector &position,
               const double radius,
               const Material &material);

        virtual Vector normal(const Vector &p) const;
        virtual double intersect(const Ray &ray) const;
};


// A point light source.
class Light {
public:
        const Vector position;
        const Colour colour;

        // Constructor.
        Light(const Vector &position,
              const Colour &colour=Colour(0xff, 0xff, 0xff));
};

// A full scene, consisting of objects (spheres) and lighting (point
// lights).
class Scene {
public:
        const std::vector<Object> objects;
        const std::vector<Light> lights;

        // Constructor.
        Scene(const std::vector<Object> &objects,
              const std::vector<Light> &lights);
};

class Renderer {
public:
        const Scene scene;

        Renderer(const Scene scene);

        // The heart of the raytracing engine.
        void render(FILE *const out) const;

private:
        const size_t width, height;

        // Trace a ray trough a given scene and output the colour.
        Colour trace(const Ray &ray,
                     Colour colour=Colour(0, 0, 0),
                     const unsigned int depth=0) const;
};

// Return the index of the object with the closest intersection, and
// the distance to the intersection `t'. If no intersection, return
// -1.
int closestIntersect(const Ray &ray, const std::vector<Object> &objects, double &t);

bool intersects(const Ray &ray, const std::vector<Object> &objects);

#endif  // _RT_H_
