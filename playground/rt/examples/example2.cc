// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

// Generated code.
rt::Renderer *getRenderer();
rt::Image    *getImage();

int main() {
        // Get generated scene and image, and render output.
        rt::render(getRenderer(), getImage(), "render2.ppm");

        return 0;
}
