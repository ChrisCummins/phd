// -*- c-basic-offset: 8; -*-
#ifndef IMAGE_H_
#define IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "./graphics.h"

// A rendered image.
class Image {
public:
    Pixel *const image;
    const size_t width;
    const size_t height;
    const size_t size;
    const Colour power;
    const bool inverted;

    Image(const size_t width, const size_t height,
          const Colour gamma = Colour(1, 1, 1),
          const bool inverted = true);

    ~Image();

    // [x,y] = value
    void inline set(const size_t x,
                    const size_t y,
                    const Colour &value) const {
        // Apply Y axis inversion if needed.
        const size_t row = inverted ? height - 1 - y : y;

        // Apply gamma correction.
        const Colour corrected = Colour(std::pow(value.r, power.r),
                                        std::pow(value.g, power.g),
                                        std::pow(value.b, power.b));

        // Explicitly cast colour to pixel data.
        image[row * width + x] = static_cast<Pixel>(corrected);
    }

    // Write data to file.
    void write(FILE *const out) const;
};

#endif  // IMAGE_H_
