// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

#include "tbb/parallel_for.h"

namespace rt {

void render(const Renderer *const renderer,
            const Image *const image,
            const char *const path) {
        const size_t numSamplesPerPixel =
                        renderer->numSubpixels * renderer->numDofSamples;

        // Print start message.
        printf("Rendering %lu pixels with %lu samples per pixel, "
               "%lu objects, and %lu light sources ...\n",
               image->size, numSamplesPerPixel,
               profiling::counters::getObjectsCount(),
               profiling::counters::getLightsCount());

        // Start timer.
        profiling::Timer t = profiling::Timer();

        // Render the scene to the output file.
        renderer->render(image);

        // Get elapsed time.
        Scalar runTime = t.elapsed();

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
        profiling::Counter traceCount = profiling::counters::getTraceCount();
        profiling::Counter rayCount   = profiling::counters::getRayCount();
        profiling::Counter traceRate  = traceCount / runTime;
        profiling::Counter rayRate    = rayCount / runTime;
        profiling::Counter pixelRate  = image->size / runTime;
        Scalar tracePerPixel = static_cast<Scalar>(traceCount)
                        / static_cast<Scalar>(image->size);

        // Print performance summary.
        printf("Rendered %lu pixels from %lu traces in %.3f seconds.\n\n",
               image->size, traceCount, runTime);
        printf("Render performance:\n");
        printf("\tRays per second:\t%lu\n", rayRate);
        printf("\tTraces per second:\t%lu\n", traceRate);
        printf("\tPixels per second:\t%lu\n", pixelRate);
        printf("\tTraces per pixel:\t%.2f\n", tracePerPixel);
}

}  // namespace rt
