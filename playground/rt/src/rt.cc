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
#include "rt/rt.h"

#include <iostream>

#include "tbb/parallel_for.h"

namespace rt {

void render(const Renderer &renderer,
            const std::string path,
            Image *const restrict image) {
        // Print start message.
        printf("Rendering %lu pixels, with "
               "%llu objects, and %llu light sources ...\n",
               image->size,
               profiling::counters::getObjectsCount(),
               profiling::counters::getLightsCount());

        // Start timer.
        profiling::Timer t = profiling::Timer();

        // Render the scene to the output file.
        renderer.render(image);

        // Get elapsed time.
        Scalar runTime = t.elapsed();

        // Open the output file.
        std::cout << "Opening file '" << path << "'..." << std::endl;
        std::ofstream out;
        out.open(path);

        // Write image to output file.
        out << *image;

        // Close the output file.
        std::cout << "Closing file '" << path << "'..." << std::endl;
        std::cout << std::endl;
        out.close();

        // Calculate performance information.
        profiling::Counter traceCount = profiling::counters::getTraceCount();
        profiling::Counter rayCount   = profiling::counters::getRayCount();
        profiling::Counter traceRate  = traceCount / runTime;
        profiling::Counter rayRate    = rayCount / runTime;
        profiling::Counter pixelRate  = image->size / runTime;
        Scalar tracePerPixel = static_cast<Scalar>(traceCount)
                        / static_cast<Scalar>(image->size);

        // Print performance summary.
        printf("Rendered %lu pixels from %llu traces in %.3f seconds.\n\n",
               image->size, traceCount, runTime);
        printf("Render performance:\n");
        printf("\tRays per second:\t%llu\n", rayRate);
        printf("\tTraces per second:\t%llu\n", traceRate);
        printf("\tPixels per second:\t%llu\n", pixelRate);
        printf("\tTraces per pixel:\t%.2f\n", tracePerPixel);
}

}  // namespace rt
