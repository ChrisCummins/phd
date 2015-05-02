// -*- c-basic-offset: 8; -*-
#ifndef RT_H_
#define RT_H_

#include <cstdint>

#include "./scene.h"
#include "./camera.h"
#include "./image.h"
#include "./ray.h"
#include "./random.h"

// A simple ray tacer. Features:
//
//   * Objects: Spheres, Planes, Checkerboards.
//   * Lighting: Point & soft lighting, reflections.
//   * Shading: Lambert (diffuse) and Phong (specular).
//   * Anti-aliasing: Stochastic supersampling.
class Renderer {
 public:
        Renderer(const Scene  *const scene,
                 const Camera *const camera,
                 const size_t maxDepth = 100,
                 const size_t aaSamples = 0,
                 const size_t aaRadius = 8);

        ~Renderer();

        const Scene *const scene;
        const Camera *const camera;

        // Renderer configuration:

        // The maximum depth to trace reflected rays to:
        const size_t maxDepth;

        // The number of *additional* samples to perform for
        // antialiasing:
        const size_t aaSamples;

        // The total number of samples = 1 + aaSamples.
        const size_t totalSamples;

        // A random distribution sampler for offseting rays in
        // stochastic anti-aliasing:
        mutable UniformDistribution aaSampler;

        // The heart of the raytracing engine.
        void render(const Image *const image) const;

 private:
        // Trace a ray trough a given scene and return the final
        // colour.
        Colour trace(const Ray &ray,
                     Colour colour = Colour(0, 0, 0),
                     const unsigned int depth = 0) const;

        // Calculate the colour of a ray through supersampling.
        Colour supersample(const Ray &ray) const;
};

#endif  // RT_H_
