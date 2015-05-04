// -*- c-basic-offset: 8; -*-
#include "rt/image.h"

namespace rt {

Image::Image(const size_t _width, const size_t _height,
             const Colour _gamma, const bool _inverted)
                : data(new Pixel[_width * _height]),
                  width(_width),
                  height(_height),
                  size(_width * _height),
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

void DataImage::downsample(const Image *const image,
                           const size_t subpixels,
                           const size_t overlap) const {
        // Calculate interpolation tile size.
        const size_t tileWidth = subpixels + 2 * overlap;
        const size_t tileSize = tileWidth * tileWidth;

        // Interpolate each pixel in image.
        for (size_t y = 0; y < image->height; y++)
                for (size_t x = 0; x < image->width; x++)
                        image->set(x, y, interpolate(x, y, subpixels,
                                                     tileWidth, tileSize));
}

Colour DataImage::interpolate(const size_t image_x,
                              const size_t image_y,
                              const size_t gridWidth,
                              const size_t tileWidth,
                              const size_t tileSize) const {
        // Convert output coordinates to data image coordinates.
        const size_t y = image_y * gridWidth;
        const size_t x = image_x * gridWidth;

        // Accumulate colour values of sub-pixels.
        Colour acc;
        for (size_t j = 0; j < tileWidth; j++) {
                for (size_t i = 0; i < tileWidth; i++) {
                        // Determine data image index.
                        const size_t index = (y + j) * width + x + i;
                        acc += data[index] / tileSize;
                }
        }

        return acc;
}

}  // namespace rt
