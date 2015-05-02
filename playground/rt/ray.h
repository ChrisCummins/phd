// -*- c-basic-offset: 8; -*-
#ifndef RAY_H_
#define RAY_H_

#include "math.h"

// A ray abstraction.
class Ray {
public:
        const Vector position, direction;

        // Construct a ray at starting position and in direction.
        inline Ray(const Vector &position, const Vector &direction)
                        : position(position), direction(direction) {}
};

#endif  // RAY_H_
