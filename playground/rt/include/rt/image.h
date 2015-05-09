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
#ifndef RT_IMAGE_H_
#define RT_IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "./graphics.h"
#include "./restrict.h"

namespace rt {

namespace image {

// Helper functions to convert 2D to 1D flat array co-ordinates, and
// vice versa.

inline size_t index(const size_t x, const size_t y, const size_t width) {
        return y * width + x;
}

inline size_t x(const size_t index, const size_t width) {
        return index % width;
}

inline size_t y(const size_t index, const size_t width) {
        return index / width;
}

}  // namespace image

// A rendered image.
class Image {
 public:
        Pixel *const restrict data;
        const size_t width;
        const size_t height;
        const size_t size;
        const Scalar saturation;
        const Colour gamma;
        const bool inverted;

        Image(const size_t width, const size_t height,
              const Scalar saturation = 1,
              const Colour gamma = Colour(1, 1, 1),
              const bool inverted = true);

        ~Image();

        // [x,y] = value
        void inline set(const size_t x,
                        const size_t y,
                        const Colour &value) const {
                // Apply Y axis inversion if needed.
                const size_t row = inverted ? height - 1 - y : y;
                // Convert 2D coordinates to flat array index.
                _set(image::index(x, row, width), value);
        }

        // [index] = value
        void inline set(const size_t index,
                        const Colour &value) const {
                const size_t x = image::x(index, width);
                const size_t y = image::y(index, width);

                set(x, y, value);
        }

        // Write data to file.
        void write(FILE *const restrict out) const;

 private:
        // Padding bytes since the "inverted" member bool is only a
        // single byte.
        char _pad[7];

        void _set(const size_t i,
                  const Colour &value) const;
};

}  // namespace rt

#endif  // RT_IMAGE_H_
