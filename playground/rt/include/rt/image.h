// -*- c-basic-offset: 8; -*-
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
