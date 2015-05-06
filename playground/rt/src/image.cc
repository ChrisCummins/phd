// -*- c-basic-offset: 8; -*-
#include "rt/image.h"

namespace rt {

Image::Image(const size_t _width, const size_t _height,
             const Scalar _saturation, const Colour _gamma,
             const bool _inverted)
                : data(new Pixel[_width * _height]),
                  width(_width),
                  height(_height),
                  size(_width * _height),
                  saturation(_saturation),
                  gamma(Colour(1 / _gamma.r, 1 / _gamma.g, 1 / _gamma.b)),
                  inverted(_inverted) {}

Image::~Image() {
        // Free pixel data.
        delete[] data;
}

void Image::write(FILE *const out) const {
        // Print PPM header.
        fprintf(out, "P3\n");                      // Magic number
        fprintf(out, "%lu %lu\n", width, height);  // Image dimensions
        fprintf(out, "%d\n", PixelColourMax);      // Max colour value

        // Iterate over each point in the image, writing pixel data.
        for (size_t i = 0; i < height * width; i++) {
                const Pixel pixel = data[i];
                fprintf(out,
                        PixelFormatString" "
                        PixelFormatString" "
                        PixelFormatString" ",
                        pixel.r, pixel.g, pixel.b);

                if (!i % width)  // Add newline at the end of each row.
                        fprintf(out, "\n");
        }
}

}  // namespace rt
