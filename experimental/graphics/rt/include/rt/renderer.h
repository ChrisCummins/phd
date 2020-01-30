/* -*-c++-*-
 *
 * Copyright (C) 2015-2020 Chris Cummins.
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

#include <array>
#include <cstdint>
#include <vector>

#ifdef USE_TBB
#include "tbb/parallel_for.h"
#endif

#include "rt/camera.h"
#include "rt/image.h"
#include "rt/random.h"
#include "rt/ray.h"
#include "rt/scene.h"

namespace rt {

class Renderer {
  // Anti-aliasing tunable knobs.
  static constexpr Scalar maxPixelDiff = 0.0000005;
  static constexpr Scalar maxSubpixelDiff = 0.008;
  static constexpr size_t maxSubpixelDepth = 3;

 public:
  Renderer(const Scene &scene, const rt::Camera *const restrict camera,
           const size_t numDofSamples = 1, const size_t maxRayDepth = 5000);

  ~Renderer();

  const Scene &scene;
  const rt::Camera *const restrict camera;

  // Renderer configuration:

  // The maximum depth to trace reflected rays to:
  const size_t maxRayDepth;

  // TODO: Super sampling anti-aliasing configuration:

  // Number of samples to make for depth of field:
  const size_t numDofSamples;

  // The heart of the raytracing engine.
  template <typename Image>
  void render(Image *const image) const;

 private:
  // Recursively supersample a region.
  Colour renderRegion(const Scalar x, const Scalar y, const Scalar regionSize,
                      const Matrix &transform, const size_t depth = 0) const;

  // Get the colour value at a single point.
  Colour renderPoint(const Scalar x, const Scalar y,
                     const Matrix &transform) const;

  // Trace a ray trough a given scene and return the final
  // colour.
  Colour trace(const Ray &ray, const unsigned int depth = 0) const;

  // Perform supersample interpolation.
  Colour interpolate(const size_t image_x, const size_t image_y,
                     const size_t dataWidth,
                     const Colour *const restrict data) const;
};

template <typename Image>
void Renderer::render(Image *const image) const {
  // Create image to camera transformation matrix.
  //
  // Create a transformation matrix to scale from image
  // space coordinates (i.e. [x,y] coordinates with
  // reference to the image size) to camera space
  // coordinates (i.e. [x,y] coordinates with reference
  // to the camera's film size).
  //
  // Scale image coordinates to camera coordinates.
  const Scale scale(camera->width / image->width,
                    camera->height / image->height, 1);
  // Offset from image coordinates to camera coordinates.
  const Translation offset(-(image->width * .5), -(image->height * .5), 0);
  const auto transformMatrix = scale * offset;

  // First, we collect a single sample for every pixel in the
  // image, plus an additional border of 1 pixel on all sides.
  const size_t borderedWidth = image->width + 2;
  const size_t borderedHeight = image->height + 2;
  const size_t borderedSize = borderedWidth * borderedHeight;
  std::vector<Colour> sampled(borderedSize);

// Collect pixel samples:
#ifdef USE_TBB
  tbb::parallel_for(static_cast<size_t>(0), sampled.size(),
                    [&](const size_t index)
#else   // USE_TBB
  for (size_t index = 0; index < sampled.size(); ++index)
#endif  // USE_TBB
                    {
                      // Get the pixel coordinates.
                      const auto x = image::x(index, borderedWidth);
                      const auto y = image::y(index, borderedWidth);

                      // Sample a point in the centre of the pixel.
                      sampled[index] =
                          renderPoint(x + .5, y + .5, transformMatrix);
                    }
#ifdef USE_TBB
  );    // NOLINT
#endif  // USE_TBB

  // Allocate memory for super-sampled image.
  std::vector<Colour> superSampled(image->size);

  // For each pixel in the image:
  for (size_t index = 0; index < image->size; index++) {
    // Get the pixel coordinates.
    const size_t x = image::x(index, image->width);
    const size_t y = image::y(index, image->width);

    // Get the previously sampled pixel value.
    Colour pixel = sampled[image::index(x + 1, y + 1, borderedWidth)];

    // Create a list of all neighbouring element indices.
    const std::array<size_t, 8> neighbour_indices = {
        image::index(x - 1, y - 1, borderedWidth),
        image::index(x, y - 1, borderedWidth),
        image::index(x + 1, y - 1, borderedWidth),
        image::index(x - 1, y, borderedWidth),
        image::index(x + 1, y, borderedWidth),
        image::index(x - 1, y + 1, borderedWidth),
        image::index(x, y + 1, borderedWidth),
        image::index(x + 1, y + 1, borderedWidth)};

    // Calculate the difference between the neighbouring
    // pixel values.
    Scalar diffSum = 0;
    for (const auto neighbour_index : neighbour_indices) {
      const auto diff = pixel.diff(sampled[neighbour_index]);
      diffSum += diff;
    }

    // If the difference is above a given threshold,
    // recursively supersample the pixel.
    if (diffSum > maxPixelDiff * neighbour_indices.size()) {
      pixel = renderRegion(x, y, 1, transformMatrix);
    }

    // Set new value.
    superSampled[index] = pixel;
  }

  // Write pixel information to image.
  for (size_t index = 0; index < image->size; index++)
    image->set(index, superSampled[index]);
}

}  // namespace rt

#endif  // RT_RENDERER_H_
