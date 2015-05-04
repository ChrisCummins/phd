// -*- c-basic-offset: 8; -*-
#ifndef RT_SCENE_H_
#define RT_SCENE_H_

#include <vector>

#include "./lights.h"
#include "./objects.h"

namespace rt {

// A full scene, consisting of objects (spheres) and lighting (point
// lights).
class Scene {
 public:
        const Objects objects;
        const Lights lights;

        // Constructor.
        inline Scene(const Objects &_objects,
                     const Lights &_lights)
                : objects(_objects), lights(_lights) {}

        inline ~Scene() {
                for (size_t i = 0; i < objects.size(); i++)
                        delete objects[i];
                for (size_t i = 0; i < lights.size(); i++)
                        delete lights[i];
        }
};

}  // namespace rt

#endif  // RT_SCENE_H_
