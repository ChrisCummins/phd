/* -*-c++-*-
 *
 * Copyright (C) 2015, 2016 Chris Cummins.
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
#ifndef RT_CAMERA_H_
#define RT_CAMERA_H_

#include "rt/math.h"
#include "rt/random.h"

namespace rt {

// A lens has a focal length and aperture setting, along with a target
// focus point.
class Lens {
 public:
  const Scalar focalLength;
  const Scalar focus;
  mutable UniformDiskDistribution<Scalar> aperture;

  inline Lens(const Scalar _focalLength, const Scalar _aperture = 1,
              const Scalar _focus = 1)
      : focalLength(_focalLength), focus(_focus), aperture(_aperture) {}
};

// A camera has a position, a target that it is pointed at, a film
// size, and a lens.
class Camera {
 public:
  const Vector position;
  const Vector direction;
  const Vector filmBack;
  const Vector right;
  const Vector up;
  const Scalar width;
  const Scalar height;
  const Lens lens;
  const Scalar focusDistance;

  inline Camera(const Vector &_position, const Vector &_lookAt,
                const Scalar _width, const Scalar _height, const Lens &_lens)
      : position(_position),
        direction((_lookAt - _position).normalise()),
        filmBack(_position -
                 (_lookAt - _position).normalise() * _lens.focalLength),
        right((_lookAt - _position).normalise() | Vector(0, 1, 0)),
        up(((_lookAt - _position).normalise() | Vector(0, 1, 0)) |
           (_lookAt - _position).normalise()),
        width(_width),
        height(_height),
        lens(_lens),
        focusDistance((_position - _lookAt).size() * _lens.focus) {}
};

}  // namespace rt

#endif  // RT_CAMERA_H_
