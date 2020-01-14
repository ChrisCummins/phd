/*
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

// Include ray tracer header.
#include "rt/rt.h"

#include <array>  // NOLINT(build/include_order)

static const size_t width = 512;
static const size_t height = 512;

int main() {
  // Create colours.
  static const rt::Colour red = rt::Colour(0xff0000);
  static const rt::Colour green = rt::Colour(0x00ff00);
  static const rt::Colour blue = rt::Colour(0x0000ff);

  // Create materials.
  const std::array<rt::Material *, 3> materials = {
      new rt::Material(red, 0, 1, .2, 10, 0),
      new rt::Material(green, 0, 1, .2, 10, 0),
      new rt::Material(blue, 0, 1, .2, 10, 0)};

  // Create objects.
  const std::array<rt::Sphere *, 3> _objects = {
      new rt::Sphere(rt::Vector(0, 50, 0), 50, materials[0]),
      new rt::Sphere(rt::Vector(50, -50, 0), 50, materials[1]),
      new rt::Sphere(rt::Vector(-50, -50, 0), 50, materials[2])};

  // Create lights.
  const std::array<rt::Light *, 2> _lights = {
      new rt::SoftLight(rt::Vector(-300, 400, -400), rt::Colour(0xffffff)),
      new rt::SoftLight(rt::Vector(300, -200, 100), rt::Colour(0x505050))};

  // Create camera.
  const rt::Camera *const restrict camera =
      new rt::Camera(rt::Vector(0, 0, -200),  // position
                     rt::Vector(0, 0, 0),     // look at
                     50, 50,                  // film width & height
                     rt::Lens(50));           // focal length

  // Create collections.
  const rt::Objects objects(_objects.begin(), _objects.end());
  const rt::Lights lights(_lights.begin(), _lights.end());

  // Create scene and renderer.
  const rt::Scene scene(objects, lights);
  const rt::Renderer renderer(scene, camera);

  rt::Image<width, height> *const image = new rt::Image<width, height>();

  // Run ray tracer.
  rt::render<rt::Image<width, height>>(renderer, "render1.ppm", image);

  delete image;

  return 0;
}
