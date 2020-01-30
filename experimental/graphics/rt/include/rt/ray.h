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
#ifndef RAY_H_
#define RAY_H_

#include "rt/math.h"

namespace rt {

// A ray abstraction.
class Ray {
 public:
  const Vector position, direction;

  // Construct a ray at starting position and in direction.
  inline Ray(const Vector &_position, const Vector &_direction)
      : position(_position), direction(_direction) {}
};

}  // namespace rt

#endif  // RAY_H_
