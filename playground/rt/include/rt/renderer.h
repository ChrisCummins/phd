// -*- c-basic-offset: 8; -*-
#ifndef RT_RENDERER_H_
#define RT_RENDERER_H_

#include <cstdint>

#include "./camera.h"
#include "./image.h"
#include "./random.h"
#include "./ray.h"
#include "./scene.h"

namespace rt {

class Renderer {
 public:
        Renderer(const Scene  *const scene,
                 const rt::Camera *const camera,
                 const size_t maxDepth  = 5000,
                 const size_t aaSamples = 8,
                 const size_t aaRadius  = .9);

        ~Renderer();

        const Scene *const scene;
        const rt::Camera *const camera;

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

}  // namespace rt

#endif  // RT_RENDERER_H_
