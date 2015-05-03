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

        inline Camera(const Vector &_position,
                      const Vector &_lookAt,
                      const Vector &_up,
                      const Scalar _width,
                      const Scalar _height,
                      const Scalar _focalLength)
                : position(_position),
                  direction((_lookAt - _position).normalise()),
                  filmBack(_position - (_lookAt - _position).normalise()
                           * _focalLength),
                  right((_lookAt - _position).normalise() | _up),
                  up(((_lookAt - _position).normalise() | _up) |
                     (_lookAt - _position).normalise()),
                  width(_width),
                  height(_height) {}
};

#endif  // CAMERA_H_
