/*
 * Copyright (C) 2015, 2016 Chris Cummins.
 *
 * This file is part of rt.
 *
 * rt is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * rt is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rt.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "rt/graphics.h"

#include <cstdlib>

namespace rt {

Colour::Colour(const HSL &hsl) {
  Scalar tmp1, tmp2, tmp3[3];
  Scalar clr[3];
  Scalar h = hsl.h / 360.0;

  if (hsl.s == 0) {
    r = hsl.l;
    g = hsl.l;
    b = hsl.l;
    return;
  }

  if (hsl.l <= 0.5)
    tmp2 = hsl.l * (1.0 + hsl.s);
  else
    tmp2 = hsl.l + hsl.s - hsl.l * hsl.s;

  tmp1 = 2.0 * hsl.l - tmp2;

  tmp3[0] = h + 1.0 / 3.0;
  tmp3[1] = h;
  tmp3[2] = h - 1.0 / 3.0;

  for (size_t i = 0; i < 3; i++) {
    if (tmp3[i] < 0)
      tmp3[i] += 1.0;
    else if (tmp3[i] > 1)
      tmp3[i] -= 1.0;

    if (6.0 * tmp3[i] < 1.0)
      clr[i] = tmp1 + (tmp2 - tmp1) * tmp3[i] * 6.0;
    else if (2.0 * tmp3[i] < 1.0)
      clr[i] = tmp2;
    else if (3.0 * tmp3[i] < 2.0)
      clr[i] = (tmp1 + (tmp2 - tmp1) * ((2.0 / 3.0) - tmp3[i]) * 6.0);
    else
      clr[i] = tmp1;
  }

  r = clr[0];
  g = clr[1];
  b = clr[2];
}

HSL::HSL(const Colour &in) {
  const Colour c = in.clampRange();
  const Scalar max = c.max();
  const Scalar min = c.min();
  const Scalar delta = c.delta();

  l = (max + min) / 2;
  s = 0;
  h = 0;

  if (max != min) {
    if (l <= 0.5)
      s = delta / (max + min);
    else
      s = delta / (2.0 - delta);

    if (c.r == max)
      h = (c.g - c.b) / delta;
    else if (c.g == max)
      h = 2.0 + (c.b - c.r) / delta;
    else if (c.b == max)
      h = 4.0 + (c.r - c.g) / delta;

    h *= 60;

    if (h < 0) h += 360.0;
  }
}

}  // namespace rt
