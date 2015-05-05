// -*- c-basic-offset: 8; -*-
#ifndef RT_IMAGE_H_
#define RT_IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "./graphics.h"

namespace rt {

// A rendered image.
class Image {
 public:
        Pixel *const data;
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
                set(row * width + x, value);
        }

        void inline set(const size_t i,
                        const Colour &value) const {
                // Apply gamma correction.
                Colour corrected = Colour(std::pow(value.r, gamma.r),
                                          std::pow(value.g, gamma.g),
                                          std::pow(value.b, gamma.b));

                // TODO: Fix strange aliasing effect as a result of
                // RGB -> HSL -> RGB conversion.
                //
                // Apply saturation.
                // HSL hsl(corrected);
                // hsl.s *= saturation;

                // Convert back to .
                // corrected = Colour(hsl);

                // Explicitly cast colour to pixel data.
                data[i] = static_cast<Pixel>(corrected);
        }

        // Write data to file.
        void write(FILE *const out) const;

 private:
        // Padding bytes since the "inverted" member bool is only a
        // single byte.
        char _pad[7];
};

// A raw data image, used for supersampling.
class DataImage {
 public:
        Colour *const data;
        const size_t width;
        const size_t height;
        const size_t size;

        inline DataImage(const size_t _width,
                         const size_t _height)
                : data(new Colour[_width * _height]),
                  width(_width), height(_height),
                  size(_width * _height) {}

        inline ~DataImage() {
                delete[] data;
        }

        // Set colour at 2D coordinates [x,y].
        void inline set(const size_t x,
                        const size_t y,
                        const Colour &value) const {
                // Convert 2D coordinates to flat array index.
                set(y * width + x, value);
        }

        // Set colour using flat array index.
        void inline set(const size_t i,
                        const Colour &value) const {
                data[i] = value;
        }

        // Downsample the DataImage to an output image, interpolating
        // using a given number of subpixels and overlap region.
        void downsample(const Image *const image,
                        const size_t subpixels,
                        const size_t overlap) const;

 private:
        Colour interpolate(const size_t image_x,
                           const size_t image_y,
                           const size_t gridWidth,
                           const size_t tileWidth,
                           const size_t tileSize) const;
};

}  // namespace rt

#endif  // RT_IMAGE_H_
