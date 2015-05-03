// -*- c-basic-offset: 8; -*-
#include "rt/rt.h"

// Generated scene code.
rt::Renderer *getRenderer();
rt::Image    *getImage();

int main() {
        rt::render(getRenderer(), getImage(), "example2.ppm");

        return 0;
}
