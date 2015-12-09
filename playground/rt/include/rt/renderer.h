/* -*- c-basic-offset: 8; -*-
 *
 * Copyright (C) 2015 Chris Cummins.
 *
 * This file is part of rt.
 *
 * rt is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * rt is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rt.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef RT_RENDERER_H_
#define RT_RENDERER_H_

#include <cstdint>

#include "rt/camera.h"
#include "rt/image.h"
#include "rt/random.h"
#include "rt/ray.h"
#include "rt/scene.h"

namespace rt {

class Renderer {
 public:
        Renderer(const Scene *const restrict scene,
                 const rt::Camera *const restrict camera,
                 const size_t numDofSamples = 1,
                 const size_t maxRayDepth   = 5000);

        ~Renderer();

        const Scene *const restrict scene;
        const rt::Camera *const restrict camera;

        // Renderer configuration:

        // The maximum depth to trace reflected rays to:
        const size_t maxRayDepth;

        // TODO: Super sampling anti-aliasing configuration:

        // Number of samples to make for depth of field:
        const size_t numDofSamples;

        // The heart of the raytracing engine.
        void render(Image *const restrict image) const;

 private:
        // Recursively supersample a region.
        Colour renderRegion(const Scalar x,
                            const Scalar y,
                            const Scalar regionSize,
                            const Matrix &transform,
                            const size_t depth = 0) const;

        // Get the colour value at a single point.
        Colour renderPoint(const Scalar x,
                           const Scalar y,
                           const Matrix &transform) const;

        // Trace a ray trough a given scene and return the final
        // colour.
        Colour trace(const Ray &ray,
                     const unsigned int depth = 0) const;

        // Perform supersample interpolation.
        Colour interpolate(const size_t image_x,
                           const size_t image_y,
                           const size_t dataWidth,
                           const Colour *const restrict data) const;
};

}  // namespace rt

#endif  // RT_RENDERER_H_
