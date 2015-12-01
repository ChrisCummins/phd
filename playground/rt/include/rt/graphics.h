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
#ifndef RT_GRAPHICS_H_
#define RT_GRAPHICS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "./math.h"

namespace rt {

/*
 * Pixels and colour types.
 */

// The output type of a single R,G,B colour component.
typedef uint8_t PixelColourType;

// The maximum value of a single R,G,B colour component.
static const uint8_t PixelColourMax = 255;

// Format string to be passed to fprintf().
#define PixelFormatString "%u"

// Transform a scalar from the range [0,1] to [0,PixelColourMax]. Note
// that this transformation may be non-linear.
PixelColourType inline scale(const Scalar x) {
        return static_cast<PixelColourType>(
            x * static_cast<Scalar>(PixelColourMax));
}

// A pixel is a trio of R,G,B components.
struct Pixel { PixelColourType r, g, b; };

// Forward declaration of HSL colour type (defined below Colour).
class HSL;

// A colour is represented by R,G,B scalars, and are mutable through
// the += and /= operators. They behave identically to Vectors.
class Colour {
 public:
        Scalar r, g, b;

        // Constructor for specifying colours as 32 bit hex
        // string. E.g. 0xff00aa.
        explicit inline Colour(const int hex)
                        : r((hex >> 16) / 255.),
                        g(((hex >> 8) & 0xff) / 255.),
                        b((hex & 0xff) / 255.) {}

        // Contructor: C = (r,g,b)
        explicit inline Colour(const float _r = 0,
                               const float _g = 0,
                               const float _b = 0)
                        : r(_r), g(_g), b(_b) {}

        // Constructor from (h,s,l) definition.
        explicit Colour(const HSL &hsl);

        // Colour addition.
        auto operator+=(const Colour &c) {
                r += c.r;
                g += c.g;
                b += c.b;
        }

        // Colour subtraction.
        auto operator-=(const Colour &c) {
                r -= c.r;
                g -= c.g;
                b -= c.b;
        }

        // Scalar division.
        auto operator/=(const Scalar x) {
                r /= x;
                g /= x;
                b /= x;
        }

        // Scalar colour multiplication.
        auto operator*(const Scalar x) const {
                return Colour(r * x, g * x, b * x);
        }

        // Scalar colour divison.
        auto operator/(const Scalar x) const {
                return Colour(r / x, g / x, b / x);
        }

        // Combination of two colours: A' = (Ar * Br, Ag * Bg, Ab * Bb)
        auto operator*(const Colour &rhs) const {
                return Colour(r * rhs.r, g * rhs.g, b * rhs.b);
        }

        // Explicit cast operation from Colour -> Pixel.
        explicit operator Pixel() const {
                return {scale(clamp(r)), scale(clamp(g)), scale(clamp(b))};
        }

        auto max() const {
                return std::max(r, std::max(g, b));
        }

        auto min() const {
                return std::min(r, std::min(g, b));
        }

        auto clampRange() const {
                return Colour(clamp(r), clamp(g), clamp(b));
        }

        auto delta() const {
                return max() - min();
        }

        // Return the sum difference between the r,g,b colour
        // components.
        auto inline diff(const Colour &rhs) const {
                return fabs(rhs.r - r) + fabs(rhs.g - g) + fabs(rhs.b - b);
        }
};

// Colour as a Hue, Saturation, Luminance set.
class HSL {
 public:
        Scalar h, s, l;

        explicit HSL(const Colour &c);
};

}  // namespace rt

#endif  // RT_GRAPHICS_H_
