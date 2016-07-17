__kernel void A(__global float* a, __global float* b, __global const float* c) {
  float d = (c[((((((4) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]);
  float e = (c[((((((3) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((3) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d;
  float f = (c[((((((7) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((7) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d +
            (c[((((((7) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e;
  float g = (c[((((((2) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((2) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d +
            (c[((((((2) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e +
            (c[((((((2) * (11)) + 7)) - 1) * (8)) + (get_global_id(0))]) * f;
  float h = (c[((((((1) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((1) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d +
            (c[((((((1) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e +
            (c[((((((1) * (11)) + 7)) - 1) * (8)) + (get_global_id(0))]) * f +
            (c[((((((1) * (11)) + 2)) - 1) * (8)) + (get_global_id(0))]) * g;
  float i = (c[((((((8) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((8) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d +
            (c[((((((8) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e;
  float j = (c[((((((6) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((6) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e +
            (c[((((((6) * (11)) + 7)) - 1) * (8)) + (get_global_id(0))]) * f +
            (c[((((((6) * (11)) + 2)) - 1) * (8)) + (get_global_id(0))]) * g;
  float k = (c[((((((9) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((9) * (11)) + 4)) - 1) * (8)) + (get_global_id(0))]) * d +
            (c[((((((9) * (11)) + 7)) - 1) * (8)) + (get_global_id(0))]) * f;
  float l = (c[((((((5) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((5) * (11)) + 3)) - 1) * (8)) + (get_global_id(0))]) * e;
  float m = (c[((((((10) * (11)) + 0)) - 1) * (8)) + (get_global_id(0))]) +
            (c[((((((10) * (11)) + 8)) - 1) * (8)) + (get_global_id(0))]) * i;

  (a[(((34) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((34) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((35) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((35) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((35) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((35) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((36) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((36) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((36) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((36) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((37) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((37) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((38) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((38) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((38) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((38) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((39) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((39) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((40) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((40) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((40) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((40) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((41) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((41) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((42) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((42) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((43) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((43) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((44) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((44) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((45) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((45) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((46) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((46) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((47) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((47) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((48) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((48) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((49) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((49) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((50) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((50) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((50) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((50) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((51) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((51) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((51) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((51) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((52) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((52) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((53) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((53) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((54) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((54) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((54) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((54) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((55) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((55) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((56) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((56) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((59) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((59) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((59) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((59) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((60) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((60) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((60) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((60) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((61) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((61) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((62) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((62) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((62) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((62) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((63) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((63) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((64) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((64) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((65) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((65) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((66) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((66) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((67) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((67) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((67) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((67) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((68) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((68) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((68) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((68) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((69) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((69) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((69) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((69) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((70) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((70) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((71) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((71) - 1) * (8)) + (get_global_id(0))]) * l;
  (b[(((72) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((72) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((73) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((73) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((74) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((74) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((75) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((75) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((76) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((76) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((77) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((77) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((80) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((80) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((81) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((81) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((82) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((82) - 1) * (8)) + (get_global_id(0))]) * l;
  (b[(((85) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((85) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((87) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((87) - 1) * (8)) + (get_global_id(0))]) * h;
  (b[(((87) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((87) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((88) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((88) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((89) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((89) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((90) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((90) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((91) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((91) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((92) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((92) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((94) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((94) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((96) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((96) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((97) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((97) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((98) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((98) - 1) * (8)) + (get_global_id(0))]) * l;
  (b[(((98) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((98) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((99) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((99) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((100) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((100) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((101) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((101) - 1) * (8)) + (get_global_id(0))]) * l;
  (a[(((105) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((105) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((106) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((106) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((107) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((107) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((108) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((108) - 1) * (8)) + (get_global_id(0))]) * e;
  (a[(((111) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((111) - 1) * (8)) + (get_global_id(0))]) * h;
  (a[(((112) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((112) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((112) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((112) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((114) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((114) - 1) * (8)) + (get_global_id(0))]) * j;
  (a[(((115) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((115) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((117) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((117) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((120) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((120) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((120) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((120) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((122) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((122) - 1) * (8)) + (get_global_id(0))]) * j;
  (a[(((123) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((123) - 1) * (8)) + (get_global_id(0))]) * j;
  (b[(((123) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((123) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((124) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((124) - 1) * (8)) + (get_global_id(0))]) * j;
  (a[(((125) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((125) - 1) * (8)) + (get_global_id(0))]) * j;
  (b[(((125) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((125) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((126) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((126) - 1) * (8)) + (get_global_id(0))]) * k;
  (b[(((130) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((130) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((132) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((132) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((133) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((133) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((134) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((134) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((134) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((134) - 1) * (8)) + (get_global_id(0))]) * j;
  (a[(((135) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((135) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((136) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((136) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((137) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((137) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((138) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((138) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((139) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((139) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((139) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((139) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((140) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((140) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((140) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((140) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((141) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((141) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((141) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((141) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((142) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((142) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((144) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((144) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((145) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((145) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((146) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((146) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((147) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((147) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((148) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((148) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((149) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((149) - 1) * (8)) + (get_global_id(0))]) * k;
  (b[(((149) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((149) - 1) * (8)) + (get_global_id(0))]) * d;
  (a[(((150) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((150) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((151) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((151) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((152) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((152) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((153) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((153) - 1) * (8)) + (get_global_id(0))]) * k;
  (a[(((154) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((154) - 1) * (8)) + (get_global_id(0))]) * k;
  (b[(((155) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((155) - 1) * (8)) + (get_global_id(0))]) * j;
  (b[(((156) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((156) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((157) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((157) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((158) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((158) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((159) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((159) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((160) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((160) - 1) * (8)) + (get_global_id(0))]) * g;
  (b[(((161) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((161) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((162) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((162) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((164) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((164) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((164) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((164) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((165) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((165) - 1) * (8)) + (get_global_id(0))]) * g;
  (a[(((166) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((166) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((166) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((166) - 1) * (8)) + (get_global_id(0))]) * j;
  (a[(((167) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((167) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((168) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((168) - 1) * (8)) + (get_global_id(0))]) * f;
  (b[(((169) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((169) - 1) * (8)) + (get_global_id(0))]) * m;
  (a[(((170) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((170) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((171) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((171) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((172) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((172) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((173) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((173) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((174) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((174) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((175) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((175) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((176) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((176) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((177) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((177) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((178) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((178) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((180) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((180) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((181) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((181) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((182) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((182) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((183) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((183) - 1) * (8)) + (get_global_id(0))]) * e;
  (b[(((183) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((183) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((184) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((184) - 1) * (8)) + (get_global_id(0))]) * i;
  (b[(((186) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((186) - 1) * (8)) + (get_global_id(0))]) * j;
  (b[(((188) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((188) - 1) * (8)) + (get_global_id(0))]) * f;
  (a[(((189) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((189) - 1) * (8)) + (get_global_id(0))]) * d;
  (b[(((190) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((190) - 1) * (8)) + (get_global_id(0))]) * m;
  (a[(((199) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((199) - 1) * (8)) + (get_global_id(0))]) * m;
  (b[(((199) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((199) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((200) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((200) - 1) * (8)) + (get_global_id(0))]) * m;
  (a[(((201) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((201) - 1) * (8)) + (get_global_id(0))]) * m;
  (b[(((201) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((201) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((202) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((202) - 1) * (8)) + (get_global_id(0))]) * m;
  (a[(((203) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((203) - 1) * (8)) + (get_global_id(0))]) * m;
  (a[(((204) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((204) - 1) * (8)) + (get_global_id(0))]) * m;
  (b[(((204) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((204) - 1) * (8)) + (get_global_id(0))]) * i;
  (a[(((205) - 1) * (8)) + (get_global_id(0))]) =
      (a[(((205) - 1) * (8)) + (get_global_id(0))]) * m;
}