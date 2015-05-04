// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

#include "rt/profiling.h"

#include "tbb/parallel_for.h"

namespace {

using rt::Colour;
using rt::Object;
using rt::Objects;
using rt::Ray;
using rt::Scalar;

// Return the index of the object with the closest intersection, and
// set the distance to the intersection `t'. If no intersection,
// return the number of number of objects (i.e. an illegal index).
size_t static inline closestIntersect(const Ray &ray,
                                      const Objects &objects,
                                      Scalar *const t) {
        // Index of, and distance to closest intersect:
        size_t index = objects.size();
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
                        index = i;
                }
        }

        return index;
}

}  // namespace

namespace rt {

Renderer::Renderer(const Scene *const _scene,
                   const rt::Camera *const _camera,
                   const size_t _subpixels,
                   const size_t _overlap,
                   const size_t _maxDepth)
                : scene(_scene), camera(_camera),
                  maxDepth(_maxDepth),
                  subpixels(_subpixels), overlap(_overlap),
                  tileSize(_subpixels + 2 * _overlap),
                  numSubpixels((_subpixels + 2 * _overlap) *
                               (_subpixels + 2 * _overlap)) {}

Renderer::~Renderer() {
        delete scene;
        delete camera;
}

void Renderer::render(const Image *const image) const {
        const size_t dataWidth = image->width * subpixels + 2 * overlap;
        const size_t dataHeight = image->height * subpixels + 2 * overlap;
        const size_t dataSize = dataWidth * dataHeight;
        Colour *const data = new Colour[dataSize];

        // Scale data coordinates to camera coordinates.
        const Scale scale(camera->width / dataWidth,
                          camera->height / dataHeight, 1);
        // Offset from data coordinates to camera coordinates.
        const Translation offset(-(dataWidth * .5),
                                 -(dataHeight * .5), 0);
        // Create combined transformation matrix.
        const Matrix transform = scale * offset;

        // For each point in the super-sampled grid:
        tbb::parallel_for(
            static_cast<size_t>(0), dataSize, [&](size_t i) {
                    // Image space coordinates.
                    const Scalar x = i % dataWidth;
                    const Scalar y = i / dataWidth;

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
                    data[i] = trace(ray);
            });

        // Iterate over each pixel in the image.
        for (size_t y = 0; y < image->height; y++)
                for (size_t x = 0; x < image->width; x++)
                        image->set(x, y, interpolate(x, y, dataWidth, data));
}

Colour Renderer::interpolate(const size_t image_x,
                             const size_t image_y,
                             const size_t dataWidth,
                             const Colour *const data) const {
        const size_t y = image_y * subpixels;
        const size_t x = image_x * subpixels;

        // Accumulate colour values of sub-pixels.
        Colour acc;
        for (size_t j = 0; j < tileSize; j++) {
                for (size_t i = 0; i < tileSize; i++) {
                        const size_t index = (y + j) * dataWidth + x + i;
                        acc += data[index] / numSubpixels;
                }
        }

        return acc;
}

Colour Renderer::trace(const Ray &ray, Colour colour,
                       const unsigned int depth) const {
        // Bump the profiling counter.
        profiling::counters::incTraceCount();

        // Determine the closet ray-object intersection (if any).
        Scalar t;
        size_t index = closestIntersect(ray, scene->objects, &t);

        // If the ray doesn't intersect any object, return.
        if (index == scene->objects.size())
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

}  // namespace rt
