// -*- c-basic-offset: 8; -*-
#ifndef RT_CAMERA_H_
#define RT_CAMERA_H_

#include "./math.h"
#include "./random.h"

namespace rt {

// A lens has a focal length and aperture setting, along with a target
// focus point.
class Lens {
 public:
        const Scalar focalLength;
        const Scalar focus;
        mutable UniformDiskDistribution aperture;

        inline Lens(const Scalar _focalLength,
                    const Scalar _aperture = 1,
                    const Scalar _focus = 1)
                : focalLength(_focalLength),
                  focus(_focus),
                  aperture(UniformDiskDistribution(_aperture)) {}
};

// A camera has a position, a target that it is pointed at, a film
// size, and a lens.
class Camera {
 public:
        const Vector position;
        const Vector direction;
        const Vector filmBack;
        const Vector right;
        const Vector up;
        const Scalar width;
        const Scalar height;
        const Lens   lens;
        const Scalar focusDistance;

        inline Camera(const Vector &_position,
                      const Vector &_lookAt,
                      const Scalar _width,
                      const Scalar _height,
                      const Lens   &_lens)
                : position(_position),
                  direction((_lookAt - _position).normalise()),
                  filmBack(_position - (_lookAt - _position).normalise()
                           * _lens.focalLength),
                  right((_lookAt - _position).normalise() | Vector(0, 1, 0)),
                  up(((_lookAt - _position).normalise() | Vector(0, 1, 0)) |
                     (_lookAt - _position).normalise()),
                  width(_width),
                  height(_height),
                  lens(_lens),
                  focusDistance((_position - _lookAt).size() * _lens.focus) {}
};

}  // namespace rt

#endif  // RT_CAMERA_H_
