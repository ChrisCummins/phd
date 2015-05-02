// -*- c-basic-offset: 8; -*-
#ifndef RT_H_
#define RT_H_

#include <stdint.h>
#include <vector>

#include "./math.h"
#include "./random.h"
#include "./graphics.h"
#include "./ray.h"

// A simple ray tacer. Features:
//
//   * Objects: Spheres, Planes, Checkerboards.
//   * Lighting: Point & soft lighting, reflections.
//   * Shading: Lambert (diffuse) and Phong (specular).
//   * Anti-aliasing: Stochastic supersampling.

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
        Material(const Colour &colour,
                 const Scalar ambient,
                 const Scalar diffuse,
                 const Scalar specular,
                 const Scalar shininess,
                 const Scalar reflectivity);
};

// A physical object that light interacts with.
class Object {
 public:
        const Vector position;

        // Constructor.
        explicit Object(const Vector &position);

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
        Plane(const Vector &origin,
              const Vector &direction,
              const Material *const material);

        virtual Vector normal(const Vector &p) const;
        virtual Scalar intersect(const Ray &ray) const;
        virtual const Material *surface(const Vector &point) const;
};

class CheckerBoard : public Plane {
 public:
        const Material *const material1;
        const Material *const material2;
        const Scalar checkerWidth;

        CheckerBoard(const Vector &origin,
                     const Vector &direction,
                     const Scalar checkerWidth,
                     const Material *const material1,
                     const Material *const material2);
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
    virtual ~Light() {}

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
               const Colour &colour = Colour(0xff, 0xff, 0xff));

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
        const Colour colour;
        const size_t samples;
        mutable UniformDistribution sampler;

        // Constructor.
        SoftLight(const Vector &position, const Scalar radius,
                  const Colour &colour = Colour(0xff, 0xff, 0xff),
                  const size_t samples = 1);

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
        ~Scene();
};

// A camera has a "film" size (width and height), and a position and a
// point of focus.
class Camera {
 public:
        const Vector position;
        const Vector direction;
        const Vector filmBack;
        const Vector right;
        const Vector up;
        const Scalar width;
        const Scalar height;

        Camera(const Vector &position,
               const Vector &lookAt,
               const Vector &up,
               const Scalar width,
               const Scalar height,
               const Scalar focalLength);
};

// A rendered image.
class Image {
 public:
        Pixel *const image;
        const size_t width;
        const size_t height;
        const size_t size;
        const Colour power;
        const bool inverted;

        Image(const size_t width, const size_t height,
              const Colour gamma = Colour(1, 1, 1),
              const bool inverted = true);
        ~Image();

        // [x,y] = value
        void inline set(const size_t x,
                        const size_t y,
                        const Colour &value) const;

        // Write data to file.
        void write(FILE *const out) const;
};

class Renderer {
 public:
        const Scene *const scene;
        const Camera *const camera;

        // Renderer configuration:

        // The maximum depth to trace reflected rays to:
        const size_t maxDepth;
        // The number of *additional* samples to perform for
        // antialiasing:
        const size_t aaSamples;
        const size_t totalSamples;
        // The random distribution sampler for calculating the offsets of
        // stochastic anti-aliasing:
        mutable UniformDistribution aaSampler;

        Renderer(const Scene *const scene,
                 const Camera *const camera,
                 const size_t maxDepth = 100,
                 const size_t aaSamples = 0,
                 const size_t aaRadius = 8);
        ~Renderer();

        // The heart of the raytracing engine.
        void render(const Image *const image) const;

 private:
        // Trace a ray trough a given scene and return the final
        // colour.
        Colour trace(const Ray &ray,
                     Colour colour = Colour(0, 0, 0),
                     const unsigned int depth = 0) const;

        // Calculate the colour of ray through supersampling.
        Colour supersample(const Ray &ray) const;
};

// Return the index of the object with the closest intersection, and
// the distance to the intersection `t'. If no intersection, return
// -1.
int closestIntersect(const Ray &ray,
                     const std::vector<const Object *> &objects,
                     Scalar *const t);

// Return whether a given ray intersects any of the objects within a
// given distance.
bool intersects(const Ray &ray, const std::vector<const Object *> &objects,
                const Scalar distance);

// Trigonometric functions accepting theta angles in degrees.
Scalar inline dsin(const Scalar theta);
Scalar inline dcos(const Scalar theta);
Scalar inline datan(const Scalar theta);

#endif  // RT_H_
