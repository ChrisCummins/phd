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
                 const size_t subpixels = 6,
                 const size_t overlap   = 1,
                 const size_t maxDepth  = 5000);

        ~Renderer();

        const Scene *const scene;
        const rt::Camera *const camera;

        // Renderer configuration:

        // The maximum depth to trace reflected rays to:
        const size_t maxDepth;

        // Super sampling anti-aliasing configuration:
        const size_t subpixels;
        const size_t overlap;

        // The heart of the raytracing engine.
        void render(const Image *const image) const;

 private:
        // Private render to function to create the supersampled
        // image.
        void render(const DataImage *const image) const;

        // Trace a ray trough a given scene and return the final
        // colour.
        Colour trace(const Ray &ray,
                     Colour colour = Colour(0, 0, 0),
                     const unsigned int depth = 0) const;

        // Perform supersample interpolation.
        Colour interpolate(const size_t image_x,
                           const size_t image_y,
                           const size_t dataWidth,
                           const Colour *const data) const;
};

}  // namespace rt

#endif  // RT_RENDERER_H_
