// -*- c-basic-offset: 8; -*-

#include <algorithm>
#include <chrono>

#include "tbb/parallel_for.h"

#include "./rt.h"

// Generated renderer and image factory:
#include "quick.rt.out"

Scene::Scene(const std::vector<const Object *> &objects,
             const std::vector<const Light *> &lights)
                : objects(objects), lights(lights) {}

Scene::~Scene() {
        for (size_t i = 0; i < objects.size(); i++)
                delete objects[i];
        for (size_t i = 0; i < lights.size(); i++)
                delete lights[i];
}

Camera::Camera(const Vector &position,
               const Vector &lookAt,
               const Vector &up,
               const Scalar width,
               const Scalar height,
               const Scalar focalLength)
                : position(position),
                  direction((lookAt - position).normalise()),
                  filmBack(position - (lookAt - position).normalise()
                           * focalLength),
                  right((lookAt - position).normalise() | up),
                  up(((lookAt - position).normalise() | up) |
                     (lookAt - position).normalise()),
                  width(width),
                  height(height) {}

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

void inline Image::set(const size_t x, const size_t y,
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

Renderer::Renderer(const Scene *const scene,
                   const Camera *const camera,
                   const size_t maxDepth,
                   const size_t aaSamples,
                   const size_t aaRadius)
                : scene(scene), camera(camera), maxDepth(maxDepth),
                  aaSamples(aaSamples), totalSamples(aaSamples + 1),
                  aaSampler(UniformDistribution(-aaRadius, aaRadius)) {}

Renderer::~Renderer() {
        delete scene;
        delete camera;
}

Colour Renderer::supersample(const Ray &ray) const {
        Colour sample = Colour();

        // Trace the origin ray.
        sample += trace(ray);

        // Accumulate extra samples, randomly distributed around x,y.
        for (size_t i = 0; i < aaSamples; i++) {
                const Scalar offsetX = aaSampler();
                const Scalar offsetY = aaSampler();

                if (offsetX < 0 || offsetX > 1)
                        printf("OFFSETX %f\n", offsetX);
                if (offsetY < 0 || offsetY > 1)
                        printf("OFFSETY %f\n", offsetY);

                const Vector origin = Vector(ray.position.x + offsetX,
                                             ray.position.y + offsetY,
                                             ray.position.z);
                sample += trace(Ray(origin, ray.direction));
        }

        // Average the accumulated samples.
        sample /= aaSamples + 1;

        return sample;
}

void Renderer::render(const Image *const image) const {
        // Scale image coordinates to camera coordinates.
        const Scale scale(camera->width / image->width,
                          camera->height / image->height, 1);
        // Offset from image coordinates to camera coordinates.
        const Translation offset(-(image->width * .5),
                                 -(image->height * .5), 0);
        // Create combined transformation matrix.
        const Matrix transform = scale * offset;

        // For each pixel in the image:
        tbb::parallel_for(
            static_cast<size_t>(0), image->size, [&](size_t i) {
                    // Image space coordinates.
                    const Scalar x = i % image->width;
                    const Scalar y = i / image->width;

                    // Convert image to camera (local) space coordinates.
                    const Vector localPosition = transform * Vector(x, y, 0);

                    // Translate camera (local) space to world space.
                    const Vector lensPoint =
                                    camera->right * localPosition.x +
                                    camera->up * localPosition.y +
                                    camera->position;

                    // Determine direction from point on lens to exposure point.
                    const Vector direction =
                                    (lensPoint - camera->filmBack).normalise();

                    // Create a ray.
                    const Ray ray = Ray(camera->filmBack, direction);

                    // Sample the ray.
                    image->set(x, y, supersample(ray));
            });
}

Colour Renderer::trace(const Ray &ray, Colour colour,
                       const unsigned int depth) const {
        // Bump the profiling counter.
        traceCounter++;

        // Determine the closet ray-object intersection.
        Scalar t;
        int index = closestIntersect(ray, scene->objects, &t);

        // If the ray doesn't intersect any object, return.
        if (index == -1)
            return colour;

        // Object with closest intersection.
        const Object *object = scene->objects[index];
        // Point of intersection.
        const Vector intersect = ray.position + ray.direction * t;
        // Surface normal at point of intersection.
        const Vector normal = object->normal(intersect);
        // Direction between intersection and source ray.
        const Vector toRay = (ray.position - intersect).normalise();
        // Material at point of intersection.
        const Material *material = object->surface(intersect);

        // Apply ambient lighting.
        colour += material->colour * material->ambient;

        // Apply shading from each light source.
        for (size_t i = 0; i < scene->lights.size(); i++)
                colour += scene->lights[i]->shade(intersect, normal, toRay,
                                                 material, scene->objects);

        // Create reflection ray and recursive evaluate.
        const Scalar reflectivity = material->reflectivity;
        if (depth < maxDepth && reflectivity > 0) {
                // Direction of reflected ray.
                const Vector reflectionDirection = (normal * 2*(normal ^ toRay)
                                                    - toRay).normalise();
                // Create a reflection.
                const Ray reflection(intersect, reflectionDirection);
                // Add reflection light.
                colour += trace(reflection, colour, depth + 1) * reflectivity;
        }

        return colour;
}

int closestIntersect(const Ray &ray,
                     const std::vector<const Object *> &objects,
                     Scalar *const t) {
        // Index of, and distance to closest intersect:
        int index = -1;
        *t = INFINITY;

        // For each object:
        for (size_t i = 0; i < objects.size(); i++) {
                // Get intersect distance.
                Scalar currentT = objects[i]->intersect(ray);

                // Check if intersects, and if so, whether the
                // intersection is closer than the current best.
                if (currentT != 0 && currentT < *t) {
                        // New closest intersection.
                        *t = currentT;
                        index = static_cast<int>(i);
                }
        }

        return index;
}

// Return the length of array.
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))
// Return the end of an array.
#define ARRAY_END(x) (x + ARRAY_LENGTH(x))

// Program entry point.
int main() {
        // Get the renderer and image.
        const Renderer *const renderer = getRenderer();
        const Image *const image = getImage();

        // Print start message.
        printf("Rendering %lu pixels with %lu samples per pixel, "
               "%lu objects, and %lu light sources ...\n",
               image->size, renderer->totalSamples,
               objectsCount, lightsCount);

        // Record start time.
        const std::chrono::high_resolution_clock::time_point startTime
                        = std::chrono::high_resolution_clock::now();

        // Render the scene to the output file.
        renderer->render(image);

        // Record end time.
        const std::chrono::high_resolution_clock::time_point endTime
                        = std::chrono::high_resolution_clock::now();

        // Open the output file.
        const char *path = "render.ppm";
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
        uint64_t traceCount = static_cast<uint64_t>(traceCounter);
        uint64_t rayCount = static_cast<uint64_t>(rayCounter);
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

        return 0;
}
