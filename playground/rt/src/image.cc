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
#include "rt/image.h"

#include <iostream>

namespace rt {

Image::Image(const size_t _width, const size_t _height,
             const Scalar _saturation, const Colour _gamma,
             const bool _inverted)
                : data(_width * _height),
                  width(_width),
                  height(_height),
                  size(_width * _height),
                  saturation(_saturation),
                  gamma(Colour(1 / _gamma.r, 1 / _gamma.g, 1 / _gamma.b)),
                  inverted(_inverted) {}

Image::~Image() {}

void Image::_set(const size_t i,
                 const Colour &value) {
        // Apply gamma correction.
        Colour corrected = Colour(std::pow(value.r, gamma.r),
                                  std::pow(value.g, gamma.g),
                                  std::pow(value.b, gamma.b));

        // TODO: Fix strange aliasing effect as a result of
        // RGB -> HSL -> RGB conversion.
        // HSL hsl(corrected);
        //
        // Apply saturation.
        // hsl.s *= saturation;
        //
        // Convert back to RGB colour.
        // corrected = Colour(hsl);

        // Explicitly cast colour to pixel data.
        data[i] = static_cast<Pixel>(corrected);
}

}  // namespace rt
