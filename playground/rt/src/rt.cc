// -*- c-basic-offset: 8; -*-
#include <chrono>

#include "rt/rt.h"

#include "tbb/parallel_for.h"

namespace rt {

void render(const Renderer *const renderer,
            const Image *const image,
            const char *const path) {
        // Print start message.
        printf("Rendering %lu pixels with %lu samples per pixel, "
               "%lu objects, and %lu light sources ...\n",
               image->size, renderer->totalSamples,
               profiling::objectsCount, profiling::lightsCount);

        // Record start time.
        const std::chrono::high_resolution_clock::time_point startTime
                        = std::chrono::high_resolution_clock::now();

        // Render the scene to the output file.
        renderer->render(image);

        // Record end time.
        const std::chrono::high_resolution_clock::time_point endTime
                        = std::chrono::high_resolution_clock::now();

        // Open the output file.
        printf("Opening file '%s'...\n", path);
        FILE *const out = fopen(path, "w");

        // Write to output file.
        image->write(out);

        // Close the output file.
        printf("Closing file '%s'...\n\n", path);
        fclose(out);

        // Free heap memory.
        delete renderer;
        delete image;

        // Calculate performance information.
        Scalar elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime).count() / 1e6;
        uint64_t traceCount = static_cast<uint64_t>(profiling::traceCounter);
        uint64_t rayCount = static_cast<uint64_t>(profiling::rayCounter);
        uint64_t traceRate = traceCount / elapsed;
        uint64_t rayRate = rayCount / elapsed;
        uint64_t pixelRate = image->size / elapsed;
        Scalar tracePerPixel = static_cast<Scalar>(traceCount)
                        / static_cast<Scalar>(image->size);

        // Print performance summary.
        printf("Rendered %lu pixels from %lu traces in %.3f seconds.\n\n",
               image->size, traceCount, elapsed);
        printf("Render performance:\n");
        printf("\tRays per second:\t%lu\n", rayRate);
        printf("\tTraces per second:\t%lu\n", traceRate);
        printf("\tPixels per second:\t%lu\n", pixelRate);
        printf("\tTraces per pixel:\t%.2f\n", tracePerPixel);
}

}  // namespace rt
