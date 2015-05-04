// -*- c-basic-offset: 8; -*-
#ifndef RT_LIGHTS_H_
#define RT_LIGHTS_H_

#include <cstdint>
#include <cstddef>
#include <vector>

#include "./math.h"
#include "./random.h"
#include "./graphics.h"
#include "./objects.h"

namespace rt {

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
                         const Objects objects) const = 0;
};

typedef std::vector<const Light *> Lights;

// A round light source.
class SoftLight : public Light {
 public:
        const Vector position;
        const Colour colour;
        const size_t samples;
        mutable UniformDistribution sampler;

        // Constructor.
        inline SoftLight(const Vector &_position,
                         const Colour &_colour = Colour(0xff, 0xff, 0xff),
                         const Scalar _radius = 0,
                         const size_t _samples = 1)
                : position(_position), colour(_colour), samples(_samples),
                           sampler(UniformDistribution(-_radius, _radius)) {
                // Register lights with profiling counter.
                profiling::counters::incLightsCount(_samples);
        }

        virtual Colour shade(const Vector &point,
                             const Vector &normal,
                             const Vector &toRay,
                             const Material *const material,
                             const Objects objects) const;
};

}  // namespace rt

#endif  // RT_LIGHTS_H_
