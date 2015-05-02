// -*- c-basic-offset: 8; -*-
#ifndef RT_H_
#define RT_H_

#include <stdint.h>
#include <vector>

#include "./math.h"
#include "./random.h"
#include "./graphics.h"
#include "./ray.h"
#include "./objects.h"
#include "./lights.h"

// A simple ray tacer. Features:
//
//   * Objects: Spheres, Planes, Checkerboards.
//   * Lighting: Point & soft lighting, reflections.
//   * Shading: Lambert (diffuse) and Phong (specular).
//   * Anti-aliasing: Stochastic supersampling.

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

#endif  // RT_H_
