// -*- c-basic-offset: 8; -*-
#ifndef RAY_H_
#define RAY_H_

#include "math.h"

namespace rt {

// A ray abstraction.
class Ray {
public:
        const Vector position, direction;

        // Construct a ray at starting position and in direction.
        inline Ray(const Vector &_position, const Vector &_direction)
                        : position(_position), direction(_direction) {}
};

}  // namespace rt

#endif  // RAY_H_
