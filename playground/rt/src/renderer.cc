// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

#include "rt/profiling.h"

#include "tbb/parallel_for.h"

namespace {

// We're using an anonymous namespace so we're allowed to import rt::
using namespace rt;  // NOLINT(build/namespaces)

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

// Create a transformation matrix to scale from image space
// coordinates (i.e. [x,y] coordinates with reference to the image
// size) to camera space coordinates (i.e. [x,y] coordinates with
// reference to the camera's film size).
Matrix cameraImageTransform(const Camera *const camera,
                            const DataImage *const image) {
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
                   const size_t _overlap,
                   const size_t _maxDepth)
                : scene(_scene), camera(_camera),
                  maxDepth(_maxDepth),
                  subpixels(_subpixels),
                  overlap(_overlap) {}

Renderer::~Renderer() {
        delete scene;
        delete camera;
}

void Renderer::render(const Image *const image) const {
        // Create an enlarged image to render to.
        const size_t width  = image->width  * subpixels + 2 * overlap;
        const size_t height = image->height * subpixels + 2 * overlap;
        const DataImage *const data = new DataImage(width, height);

        // Render to this image.
        render(data);

        // Shrink this larger image to the final output size.
        data->downsample(image, subpixels, overlap);

        delete data;
}

void Renderer::render(const DataImage *const image) const {
        // Create combined transformation matrix.
        const Matrix transform = cameraImageTransform(camera, image);

        // For each pixel in the image:
        tbb::parallel_for(
            static_cast<size_t>(0), image->size, [&](size_t i) {
                    // Get the image space coordinates.
                    const Scalar x = i % image->width;
                    const Scalar y = i / image->width;

                    // Convert image to camera space coordinates.
                    const Vector localPosition = transform * Vector(x, y, 0);

                    // Translate camera space to world space.
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
                    image->set(i, trace(ray));
            });
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
