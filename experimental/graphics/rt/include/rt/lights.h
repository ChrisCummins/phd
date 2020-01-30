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
#ifndef RT_LIGHTS_H_
#define RT_LIGHTS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rt/graphics.h"
#include "rt/math.h"
#include "rt/objects.h"
#include "rt/random.h"
#include "rt/restrict.h"

namespace rt {

// Base class light source.
class Light {
 public:
  // Virtual destructor.
  virtual ~Light() {}

  // Calculate the shading colour at `point' for a given surface
  // material, surface normal, and direction to the ray.
  virtual Colour shade(const Vector &point, const Vector &normal,
                       const Vector &toRay,
                       const Material *const restrict material,
                       const Objects objects) const = 0;
};

using Lights = const std::vector<Light *>;

// A round light source.
class SoftLight : public Light {
 public:
  const Vector position;
  const Colour colour;
  const size_t samples;
  mutable UniformDistribution<Scalar> sampler;

  // Constructor.
  inline SoftLight(const Vector &_position,
                   const Colour &_colour = Colour(0xff, 0xff, 0xff),
                   const Scalar _radius = 0, const size_t _samples = 1)
      : position(_position),
        colour(_colour),
        samples(_samples),
        sampler(-_radius, _radius) {
    // Register lights with profiling counter.
    profiling::counters::incLightsCount(_samples);
  }

  virtual Colour shade(const Vector &point, const Vector &normal,
                       const Vector &toRay,
                       const Material *const restrict material,
                       const Objects objects) const;
};

}  // namespace rt

#endif  // RT_LIGHTS_H_
