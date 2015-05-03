// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

#include "tbb/parallel_for.h"

namespace {

// Return the index of the object with the closest intersection, and
// the distance to the intersection `t'. If no intersection, return
// the number of number of objects (i.e. an illegal index).
size_t closestIntersect(const Ray &ray,
                        const std::vector<const Object *> &objects,
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

Renderer::Renderer(const Scene *const _scene,
                   const Camera *const _camera,
                   const size_t _maxDepth,
                   const size_t _aaSamples,
                   const size_t _aaRadius)
                : scene(_scene), camera(_camera), maxDepth(_maxDepth),
                  aaSamples(_aaSamples), totalSamples(_aaSamples + 1),
                  aaSampler(UniformDistribution(-_aaRadius, _aaRadius)) {}

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
