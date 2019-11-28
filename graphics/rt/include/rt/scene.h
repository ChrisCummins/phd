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
#ifndef RT_SCENE_H_
#define RT_SCENE_H_

#include <vector>

#include "rt/lights.h"
#include "rt/objects.h"

namespace rt {

// A full scene, consisting of objects (spheres) and lighting (point
// lights).
class Scene {
 public:
  const Objects objects;
  const Lights lights;

  // Constructor.
  inline Scene(const Objects &_objects, const Lights &_lights)
      : objects(_objects), lights(_lights) {}

  inline ~Scene() {
    for (auto object : objects) delete object;
    for (auto light : lights) delete light;
  }
};

}  // namespace rt

#endif  // RT_SCENE_H_
