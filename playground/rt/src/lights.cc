// -*- c-basic-offset: 8; -*-
#include "rt/lights.h"

#include "rt/profiling.h"

namespace rt {

namespace {

// Return whether a given ray intersects any of the objects within a
// given distance.
bool intersects(const Ray &ray, const std::vector<const Object *> &objects,
                const Scalar distance) {
    // Determine any object intersects ray within distance:
    for (size_t i = 0; i < objects.size(); i++) {
        const Scalar t = objects[i]->intersect(ray);
        if (t > 0 && t < distance)
            return true;
    }

    // No intersect.
    return false;
}

}  // namespace

Colour PointLight::shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const {
        // Shading is additive, starting with black.
        Colour output = Colour();

        // Vector from point to light.
        const Vector toLight = position - point;
        // Distance from point to light.
        const Scalar distance = toLight.size();
        // Direction from point to light.
        const Vector direction = toLight / distance;

        // Determine whether light is blocked.
        const bool blocked = intersects(Ray(point, direction),
                                        objects, distance);
        // Do nothing without line of sight.
        if (blocked)
                return output;

        profiling::counters::incRayCount();

        // Product of material and light colour.
        const Colour illumination = colour * material->colour;

        // Apply Lambert (diffuse) shading.
        const Scalar lambert = std::max(normal ^ direction,
                                        static_cast<Scalar>(0));
        output += illumination * material->diffuse * lambert;

        // Apply Blinn-Phong (specular) shading.
        const Vector bisector = (toRay + direction).normalise();
        const Scalar phong = pow(std::max(normal ^ bisector,
                                          static_cast<Scalar>(0)),
                                 material->shininess);
        output += illumination * material->specular * phong;

        return output;
}

SoftLight::SoftLight(const Vector &_position, const Scalar _radius,
                     const Colour &_colour, const size_t _samples)
                : position(_position), colour(_colour), samples(_samples),
                  sampler(UniformDistribution(-_radius, _radius)) {
        // Register lights with profiling counter.
        profiling::counters::incLightsCount(_samples);
}

Colour SoftLight::shade(const Vector &point,
                        const Vector &normal,
                        const Vector &toRay,
                        const Material *const material,
                        const std::vector<const Object *> objects) const {
        // Shading is additive, starting with black.
        Colour output = Colour();

        // Product of material and light colour.
        const Colour illumination = (colour * material->colour) / samples;

        // Cast multiple light rays, nomrally distributed about the
        // light's centre.
        for (size_t i = 0; i < samples; i++) {
                // Create a new point origin randomly offset from centre.
                const Vector origin = Vector(position.x + sampler(),
                                             position.y + sampler(),
                                             position.z + sampler());
                // Vector from point to light.
                const Vector toLight = origin - point;
                // Distance from point to light.
                const Scalar distance = toLight.size();
                // Direction from point to light.
                const Vector direction = toLight / distance;

                // Determine whether light is blocked.
                const bool blocked = intersects(Ray(point, direction),
                                                objects, distance);
                // Do nothing without line of sight.
                if (blocked)
                        continue;

                // Bump the profiling counter.
                profiling::counters::incRayCount();

                // Apply Lambert (diffuse) shading.
                const Scalar lambert = std::max(normal ^ direction,
                                                static_cast<Scalar>(0));
                output += illumination * material->diffuse * lambert;

                // Apply Blinn-Phong (specular) shading.
                const Vector bisector = (toRay + direction).normalise();
                const Scalar phong = pow(std::max(normal ^ bisector,
                                                  static_cast<Scalar>(0)),
                                         material->shininess);
                output += illumination * material->specular * phong;
        }

        return output;
}


}  // namespace rt
