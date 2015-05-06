// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

#include "rt/profiling.h"

#include "tbb/parallel_for.h"

namespace {

// We're using an anonymous namespace so we're allowed to import rt::
using namespace rt;  // NOLINT(build/namespaces)

// Return the object with the closest intersection to ray, and set the
// distance to the intersection `t'. If no intersection, returns a
// nullptr.
static inline const Object *closestIntersect(const Ray &ray,
                                             const Objects &objects,
                                             Scalar *const t) {
        // Index of, and distance to closest intersect:
        const Object *closest = nullptr;
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
                        closest = objects[i];
                }
        }

        return closest;
}

// Create a transformation matrix to scale from image space
// coordinates (i.e. [x,y] coordinates with reference to the image
// size) to camera space coordinates (i.e. [x,y] coordinates with
// reference to the camera's film size).
Matrix cameraImageTransform(const Camera *const camera,
                            const Image *const image) {
        // Scale image coordinates to camera coordinates.
        const Scale scale(camera->width / image->width,
                          camera->height / image->height, 1);
        // Offset from image coordinates to camera coordinates.
        const Translation offset(-(image->width * .5),
                                 -(image->height * .5), 0);
        // Combine the transformation matrices.
        return scale * offset;
}

}  // namespace

namespace rt {

Renderer::Renderer(const Scene *const _scene,
                   const rt::Camera *const _camera,
                   const size_t _subpixels,
                   const size_t _numDofSamples,
                   const size_t _maxDepth)
                : scene(_scene), camera(_camera),
                  maxDepth(_maxDepth),
                  subpixels(_subpixels),
                  numSubpixels(_subpixels * _subpixels),
                  subpixelWidth(1.0 / _subpixels),
                  numDofSamples(_numDofSamples) {}

Renderer::~Renderer() {
        delete scene;
        delete camera;
}

void Renderer::render(const Image *const image) const {
        // Create image to camera transformation matrix.
        const Matrix transform = cameraImageTransform(camera, image);

        // For each pixel in the image:
        tbb::parallel_for(
            static_cast<size_t>(0), image->size, [&](size_t index) {
                    // Get the pixel coordinates.
                    const Scalar x = index % image->width;
                    const Scalar y = index / image->width;

                    // Render a region of size 1x1:
                    const Colour output = renderRegion(x, y, 1, 1, transform);

                    // Set output image colour.
                    image->set(x, y, output);
            });
}

Colour Renderer::renderRegion(const Scalar regionX,
                              const Scalar regionY,
                              const Scalar regionWidth,
                              const Scalar regionHeight,
                              const Matrix &transform) const {
        Colour output;

        // Perform super-sampling by rendering multiple
        for (size_t j = 0; j < numSubpixels; j++) {
                for (size_t i = 0; i < numSubpixels; i++) {
                        const Scalar x = regionX + i * subpixelWidth;
                        const Scalar y = regionY + j * subpixelWidth;

                        output += renderPoint(x, y, transform) / numSubpixels;
                }
        }

        return output;
}

Colour Renderer::renderPoint(const Scalar x,
                             const Scalar y,
                             const Matrix &transform) const {
        Colour output;

        // Convert image to camera space coordinates.
        const Vector imageOrigin = transform * Vector(x, y, 0);

        // Translate camera space to world space.
        const Vector focalOrigin =
                        camera->right * imageOrigin.x +
                        camera->up * imageOrigin.y +
                        camera->position * 1;

        // Determine direction from point on lens to
        // exposure point.
        const Vector focalDirection =
                        (focalOrigin - camera->filmBack)
                        .normalise();

        // Determine the focus point of the pixel.
        const Vector focalPoint = camera->filmBack + focalDirection
                        * camera->focusDistance;

        // Accumulate numDofSamples samples.
        for (size_t i = 0; i < numDofSamples; i++) {
                // Convert image to camera space coordinates.
                const Vector cameraSpace = imageOrigin +
                                camera->lens.aperture();

                // Translate camera space to world space.
                const Vector worldSpace =
                                camera->right * cameraSpace.x +
                                camera->up * cameraSpace.y +
                                camera->position;

                // Determine direction from point on lens
                // to focus point.
                const Vector direction =
                                (focalPoint - worldSpace)
                                .normalise();

                // Create a ray.
                const Ray ray = Ray(worldSpace, direction);

                // Sample the ray.
                output += trace(ray) / numDofSamples;
        }

        return output;
}

Colour Renderer::trace(const Ray &ray,
                       const unsigned int depth) const {
        Colour colour;

        // Bump profiling counter.
        profiling::counters::incTraceCount();

        // Determine the closet ray-object intersection (if any).
        Scalar t;
        const Object *const object = closestIntersect(ray, scene->objects, &t);
        // If the ray doesn't intersect any object, do nothing.
        if (object == nullptr)
                return colour;

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
                colour += trace(reflection, depth + 1) * reflectivity;
        }

        return colour;
}

}  // namespace rt
