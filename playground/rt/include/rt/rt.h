// -*- c-basic-offset: 8; -*-
#ifndef RT_RT_H_
#define RT_RT_H_

#include "./image.h"
#include "./renderer.h"
#include "./restrict.h"

// A simple ray tacer. Features:
//
//   * Objects: Spheres, Planes, Checkerboards.
//   * Lighting: Point & soft lighting, reflections.
//   * Shading: Lambert (diffuse) and Phong (specular).
//   * Anti-aliasing: Stochastic supersampling.
namespace rt {

        // Render the target image and write output to path. Prints
        // profiling information.
        void render(const Renderer *const restrict renderer,
                    const Image *const restrict image,
                    const char *const restrict path);

}  // namespace rt

#endif  // RT_RT_H_
