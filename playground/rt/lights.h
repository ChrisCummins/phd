// -*- c-basic-offset: 8; -*-
#ifndef LIGHTS_H_
#define LIGHTS_H_

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "./math.h"
#include "./random.h"
#include "./graphics.h"
#include "./objects.h"

// A profiling counter that keeps track of how many times we've called
// Renderer::trace().
extern std::atomic<uint64_t> traceCounter;

// A profiling counter that keeps track of how many times we've
// contributed light to a ray.
extern std::atomic<uint64_t> rayCounter;

// Profiling counter.
extern uint64_t lightsCount;

// Base class light source.
class Light {
 public:
    // Virtual destructor.
    virtual ~Light() {}

    // Calculate the shading colour at `point' for a given surface
    // material, surface normal, and direction to the ray.
    virtual Colour shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const = 0;
};

// A point light source.
class PointLight : public Light {
 public:
    const Vector position;
    const Colour colour;

    // Constructor.
    inline PointLight(const Vector &_position,
                      const Colour &_colour = Colour(0xff, 0xff, 0xff))
            : position(_position), colour(_colour) {
        // Register light with profiling counter.
        lightsCount += 1;
    }

    virtual Colour shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const;
};

// A round light source.
class SoftLight : public Light {
 public:
        const Vector position;
        const Colour colour;
        const size_t samples;
        mutable UniformDistribution sampler;

        // Constructor.
        SoftLight(const Vector &position, const Scalar radius,
                  const Colour &colour = Colour(0xff, 0xff, 0xff),
                  const size_t samples = 1);

        virtual Colour shade(const Vector &point,
                             const Vector &normal,
                             const Vector &toRay,
                             const Material *const material,
                             const std::vector<const Object *> objects) const;
};

#endif  // LIGHTS_H_
