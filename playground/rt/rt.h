#ifndef _RT_H_
#define _RT_H_

#include <cstdio>
#include <stdint.h>
#include <vector>

// A pixel is a trio of R,G,B bytes.
struct Pixel { uint8_t r, g, b; };

// Colour class. Note that colours are NOT immutable.
class Colour {
public:
        float r, g, b;

        // Colour constructors.
        Colour(const int hex);
        Colour(const float r=0, const float g=0, const float b=0);

        // Colour addition.
        void operator+=(const Colour &c);

        // Scalar colour multiplication.
        Colour operator*(const double x) const;

        // Combination of two colours.
        Colour operator*(const Colour c) const;

        // Cast operation from Colour -> Pixel.
        operator Pixel() const;
};

// Properties that describe a material.
class Material {
public:
        const Colour colour;
        const double ambient;      // 0 <= ambient <= 1
        const double diffuse;      // 0 <= diffuse <= 1
        const double specular;     // 0 <= specular <= 1
        const double shininess;    // shininess >= 0
        const double reflectivity; // 0 <= reflectivity < 1

        // Constructor.
        Material(const Colour &colour,
                 const double ambient,
                 const double diffuse,
                 const double specular,
                 const double shininess,
                 const double reflectivity);
};

// Vector class.
class Vector {
public:
        const double x;
        const double y;
        const double z;

        // Constructors.
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
        double size() const;

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
        const Vector position, direction;

        // Constructors.
        Ray(const double x=0, const double y=0);
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
        virtual double intersect(const Ray &ray) const = 0;
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
        virtual double intersect(const Ray &ray) const;
        virtual const Material *surface(const Vector &point) const;
};

class CheckerBoard : public Plane {
public:
        const Material *const black;
        const Material *const white;
        const double checkerWidth;

        CheckerBoard(const Vector &origin,
                     const Vector &direction,
                     const double checkerWidth);
        ~CheckerBoard();

        virtual const Material *surface(const Vector &point) const;
};

// A sphere consits of a position and a radius.
class Sphere : public Object {
public:
        const double radius;
        const Material *const material;

        // Constructor.
        Sphere(const Vector &position,
               const double radius,
               const Material *const material);

        virtual Vector normal(const Vector &p) const;
        virtual double intersect(const Ray &ray) const;
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
};

// Return the index of the object with the closest intersection, and
// the distance to the intersection `t'. If no intersection, return
// -1.
int closestIntersect(const Ray &ray,
                     const std::vector<const Object *> &objects,
                     double &t);

// Return whether a given ray intersects any of the objects.
bool intersects(const Ray &ray, const std::vector<const Object *> &objects);

// Clamp a value to within the range [0,255]
uint8_t inline clamp(const double x);

#endif  // _RT_H_
