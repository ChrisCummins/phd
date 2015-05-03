// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

int main() {
        // Create objects.
        const Object *_objects[] = {
                new Sphere(Vector(0, 0, 0), 100, new Material(Colour(0xffffff),
                                                              0, 1, .2, 10, 0))
        };

        // Create lights.
        const Light *_lights[] = {
                new PointLight(Vector(-300,  300, -500), Colour(0xff0000)),
                new PointLight(Vector( 200, -200,    0), Colour(0x040710))
        };

        const std::vector<const Object *> objects(_objects, _objects + 1);
        const std::vector<const Light *>  lights( _lights,  _lights  + 2);

        // Create renderer.
        const Renderer *const renderer =
                        new Renderer(new Scene(objects, lights),
                                     new Camera(Vector(0, 0, -250),  // position
                                                Vector(0, 0, 0),     // look at
                                                Vector(0, 1, 0),     // up
                                                50, 50,  // film width & height
                                                50));    // lens focal length);

        // Run ray tracer.
        rt::render(renderer, new Image(512, 512), "example1.ppm");

        return 0;
}
