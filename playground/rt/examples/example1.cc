// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

int main() {
        const rt::Colour red   = rt::Colour(0xff0000);
        const rt::Colour green = rt::Colour(0x00ff00);
        const rt::Colour blue  = rt::Colour(0x0000ff);

        // Create objects.
        const rt::Object *_objects[] = {
                new rt::Sphere(rt::Vector(0,    50, 0), 50,
                               new rt::Material(red, 0, 1, .2, 10, 0)),
                new rt::Sphere(rt::Vector(50,  -50, 0), 50,
                               new rt::Material(green, 0, 1, .2, 10, 0)),
                new rt::Sphere(rt::Vector(-50, -50, 0), 50,
                               new rt::Material(blue, 0, 1, .2, 10, 0))
        };

        // Create lights.
        const rt::Light *_lights[] = {
                new rt::SoftLight(rt::Vector(-300,  400, -400),
                                  rt::Colour(0xffffff)),
                new rt::SoftLight(rt::Vector( 300, -200,  100),
                                  rt::Colour(0x505050))
        };

        // Create camera.
        const rt::Camera *const camera =
                        new rt::Camera(rt::Vector(0, 0, -250),  // position
                                       rt::Vector(0, 0, 0),     // look at
                                       50, 50,         // film width & height
                                       rt::Lens(50));  // focal length

        // Create collections.
        const rt::Objects objects(_objects, _objects + 3);
        const rt::Lights  lights( _lights,  _lights  + 2);

        // Create renderer.
        const rt::Renderer *const renderer =
                       new rt::Renderer(new rt::Scene(objects, lights), camera);

        // Run ray tracer.
        rt::render(renderer, new rt::Image(512, 512), "render1.ppm");

        return 0;
}
