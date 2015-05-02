// -*- c-basic-offset: 8; -*-
#ifndef CAMERA_H_
#define CAMERA_H_

#include "./math.h"

// A camera has a "film" size (width and height), and a position and a
// point of focus.
class Camera {
public:
        const Vector position;
        const Vector direction;
        const Vector filmBack;
        const Vector right;
        const Vector up;
        const Scalar width;
        const Scalar height;

        inline Camera(const Vector &position,
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
};

#endif  // CAMERA_H_
