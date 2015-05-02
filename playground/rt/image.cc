// -*- c-basic-offset: 8; -*-
#include "./image.h"

Image::Image(const size_t width, const size_t height,
             const Colour gamma, const bool inverted)
                : image(new Pixel[width * height]),
                  width(width), height(height), size(width * height),
                  power(Colour(1 / gamma.r, 1 / gamma.g, 1 / gamma.b)),
                  inverted(inverted) {}

Image::~Image() {
        // Free pixel data.
        delete[] image;
}

void Image::write(FILE *const out) const {
        // Print PPM header.
        fprintf(out, "P3\n");                      // Magic number
        fprintf(out, "%lu %lu\n", width, height);  // Image dimensions
        fprintf(out, "%d\n", PixelColourMax);      // Max colour value

        // Iterate over each point in the image, writing pixel data.
        for (size_t i = 0; i < height * width; i++) {
                const Pixel pixel = image[i];
                fprintf(out,
                        PixelFormatString" "
                        PixelFormatString" "
                        PixelFormatString" ",
                        pixel.r, pixel.g, pixel.b);

                if (!i % width)  // Add newline at the end of each row.
                        fprintf(out, "\n");
        }
}
