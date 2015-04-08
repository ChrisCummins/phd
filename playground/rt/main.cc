#include <algorithm>
#include <math.h>
#include <cstdio>
#include <stdint.h>
#include <cassert>
#include <vector>


// A pixel is a trio of R,G,B bytes.
struct Pixel { uint8_t r, g, b; };


// Clamp a value to within the range [0,255].
uint8_t clamp(const double x) {
  const double min = 0;
  const double max = 255;

  return static_cast<uint8_t>(std::max (std::min(x, max), min));
}

// Colour class.
struct Colour {
  // Colour components.
  float r, b, g;

  // Colour constructor.
  Colour(const float r=0, const float g=0, const float b=0) {
    this->r = r;
    this->g = g;
    this->b = b;
  }

  // Colour addition.
  void operator+=(const Colour &c) {
    r += c.r;
    g += c.g;
    b += c.b;
  }

  // Scalar colour multiplication.
  Colour operator*(const double x) const {
    return Colour(r * x, g * x, b * x);
  }

  // Combination of two colours.
  Colour operator*(const Colour c) const {
    return Colour(r * (c.r / 255), g * (c.g / 255), b * (c.b / 255));
  }

  // Explicit cast operation for Colour -> Pixel.
  explicit operator Pixel() const {
    return {clamp(r), clamp(g), clamp(b)};
  }

#ifdef DEBUG
  // Print to stdout: #rrggbb
  void print() const {
    printf("Colour[%p] #%02x%02x%02x\n", this,
           clamp(r), clamp(g), clamp(b));
  }

  // Returns true if value is equal to r, g, b literals.
  bool eq(const float r, const float g, const float b,
          bool verbose=true) {
    bool equal = this->r == r && this->g == g && this->b == b;

    if (verbose && !equal)
      print();

    return equal;
  }
#endif
};



// Properties that describe a material.
struct Material {
  const Colour diffuse;
  const double diffuseCoefficient;

  Material(const Colour &diffuse, const double diffuseCoefficient)
      : diffuse(diffuse), diffuseCoefficient(diffuseCoefficient) {}
};



// Vector class.
struct Vector {
  // Vector components.
  double x, y, z;

  // Vector constructor.
  Vector(const double x=0, const double y=0, const double z=0) {
    this->x=x;
    this->y=y;
    this->z=z;
  }

  // Vector addition.
  Vector operator+(const Vector &b) const {
    return Vector(x + b.x, y + b.y, z + b.z);
  }

  // Vector subtraction.
  Vector operator-(const Vector &b) const {
    return Vector(x - b.x, y - b.y, z - b.z);
  }

  // Scalar multiplication.
  Vector operator*(const double a) const {
    return Vector(a * x, a * y, a * z);
  }

  // Scalar division.
  Vector operator/(const double a) const {
    return Vector(x / a, y / a, z / a);
  }

  // Vector product.
  Vector operator*(const Vector &b) const {
    return Vector(x * b.x, y * b.y, z * b.z);
  }

  // Equality testing.
  bool operator==(const Vector &b) const {
    return x == b.x && y == b.y && z == b.z;
  }

  // Negative equality testing.
  bool operator!=(const Vector &b) const {
    return !(*this == b);
  }

  // Length of vector.
  double magnitude() const {
    return sqrt(x * x + y * y + z * z);
  }

  // Scalar product of components.
  double product() const {
    return x * y * z;
  }

  // Scalar sum of components.
  double sum() {
    return x + y + z;
  }

  // Normalise vector.
  Vector normalise() const {
    return *this / magnitude();
  }

#ifdef DEBUG
  // Returns true if value is equal to x, y, z literals.
  bool eq(const double x, const double y, const double z,
          bool verbose=true) {
    bool equal = this->x == x && this->y == y && this->z == z;

    if (verbose && !equal)
      print();

    return equal;
  }

  // Print to stdout: {x, y, z}
  void print() const {
    printf("Vector[%p] {%.1f %.1f %.1f}\n", this, x, y, z);
  }
#endif
};

// Vector dot product.
double dot(const Vector &a, const Vector &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Vector cross product.
Vector cross(const Vector &a, const Vector &b) {
  return Vector(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.z);
}

// Starting depth of rays.
static const double RAY_START_Z = -1000;
// We use this value to accomodate for rounding errors in the
// intersect() code.
static const double ROUNDING_ERROR = 1e-6;



// A sphere consits of a position and a radius.
struct Sphere {
  const Vector position;
  const double radius;
  const Material material;

  Sphere(const Vector &position,
         const double radius,
         const Material &material)
      : position(position), radius(radius), material(material) {}

  // Return surface normal at point p.
  Vector surfaceNormal(const Vector &p) const {
    return (p - position).normalise();
  }

#ifdef DEBUG
  // Returns true if values are equal.
  bool eq(const double x, const double y, const double z, const double r,
          bool verbose=true) {
    bool equal = position.x == x && position.y == y && position.z == z && radius == r;

    if (verbose && !equal)
      print();

    return equal;
  }

  // Print to stdout.
  void print() const {
    printf("Sphere[%p] {%.1f %.1f %.1f} %.1f\n", this,
           position.x, position.y, position.z, radius);
  }
#endif
};



// A ray consists of a position and a (normalised) direction.
struct Ray {
  Vector position, direction;

  Ray(const double x=0, const double y=0) {
    position = Vector(x, y, RAY_START_Z);
    direction = Vector(0, 0, 1);
  }

  Ray(const Vector &position, const Vector &direction) {
    this->position = position;
    this->direction = direction;
  }

  // Return the distance to intersect of the given sphere. If no
  // intersection, return 0.
  double intersect(const Sphere &s) const {
    const Vector distance = s.position - position;
    const double B = dot(direction, distance);
    const double D = B * B - dot(distance, distance) + s.radius * s.radius;

    // No solution.
    if (D < 0)
      return 0;

    const double t0 = B - sqrt(D);
    const double t1 = B + sqrt(D);

    if (t0 > ROUNDING_ERROR)
      return t0;
    else if (t1 > ROUNDING_ERROR)
      return t1;
    else
      return 0;
  }

  int closestIntersect(const std::vector<Sphere> &spheres, double &t) const {
    int index = -1;
    t = INFINITY; // Distance to closest intersect.

    // For each object:
    for (size_t i = 0; i < spheres.size(); i++) {
      const Sphere sphere = spheres[i];
      double currentT = intersect(sphere);

      // Check if intersects, and if so, whether the intersection is
      // closer than the current best.
      if (currentT > 0 && currentT < t) {
        // New closest intersection.
        t = currentT;
        index = static_cast<int>(i);
      }
    }

    return index;
  }

  bool intersects(const std::vector<Sphere> &spheres) const {
    for (std::vector<Sphere>::const_iterator i = spheres.begin();
         i != spheres.end(); i++) {
      double t = intersect(*i);
      if (t > 0) {
        return true;
      }
    }
    return false;
  }

#ifdef DEBUG
  // Returns true if value is equal to x, y, z literals.
  bool eq(const double x, const double y, const double z,
          const double dx, const double dy, const double dz,
          bool verbose=true) const {
    bool equal = (position.x == x && position.y == y && position.z == z &&
                  direction.x == dx && direction.y == dy && direction.z == dz);

    if (verbose && !equal)
      print();

    return equal;
  }

  // Print to stdout.
  void print() const {
    printf("Ray[%p] {%.1f %.1f %.1f} -> {%.1f %.1f %.1f}\n", this,
           position.x, position.y, position.z,
           direction.x, direction.y, direction.z);
  }
#endif
};



// A point light source.
struct Light {
  const Vector position;
  const Colour colour;

  // Constructor.
  Light(const Vector &position, const Colour &colour=Colour(0xaa, 0xaa, 0xaa))
      : position(position), colour(colour) {};

#ifdef DEBUG
  // Print to stdout.
  void print() const {
    printf("Light[%p] {%.1f %.1f %.1f} #%02x%02x%02x\n", this,
           position.x, position.y, position.z,
           clamp(colour.r), clamp(colour.g), clamp(colour.b));
  }
#endif
};


// The maximum depth to trace rays for.
static const unsigned int MAX_DEPTH = 1;


// A full scene, consisting of objects (spheres) and lighting (point
// lights).
struct Scene {
  const std::vector<Sphere> spheres;
  const std::vector<Light> lights;

  // Constructor.
  Scene(const std::vector<Sphere> &spheres, const std::vector<Light> &lights)
      : spheres(spheres), lights(lights) {}

  // Trace a ray and set the colour.
  void traceRay(const Ray &ray, Colour &colour, const unsigned int depth=0) const {
    // Do nothing if we have reached the maximum depth.
    if (depth > MAX_DEPTH)
      return;

    // Determine the closet ray-object intersection.
    double t;
    int index = ray.closestIntersect(spheres, t);

    if (index != -1) {
      // Object with closest intersection.
      const Sphere sphere = spheres[index];
      // Point of intersection.
      const Vector intersect = ray.position + ray.direction * t;
      // Surface normal at point of intersection.
      const Vector surfaceNormal = sphere.surfaceNormal(intersect);

      // Accumulate each light in turn:
      for (std::vector<Light>::const_iterator l = lights.begin();
           l != lights.end(); l++) {
        const Light light = *l;
        // Direction vector from intersection to light.
        const Vector toLight = (light.position - intersect).normalise();
        const Ray shadowRay = Ray(intersect, toLight);
        const bool blocked = shadowRay.intersects(spheres);

        // Don't apply shading if the light is blocked.
        if (!blocked) {
          // Diffuse lighting.
          const Colour illumination = light.colour * sphere.material.diffuse;
          const double lambert = std::max(dot(surfaceNormal, toLight),
                                          static_cast<double>(0));

          colour += illumination * sphere.material.diffuseCoefficient * lambert;
        }
      }
    }
  }

  // The heart of the raytracing engine.
  void render(const size_t width, const size_t height, FILE *const out) const {
    printf("Rendering scene size [%lu x %lu] ...\n", width, height);

    Pixel image[height][width];

    // For each pixel in the screen:
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        Colour colour = Colour(0, 0, 0);
        Ray ray(x, y);

        // Trace the ray.
        traceRay(ray, colour);

        // Convert final colour to pixel data.
        image[y][x] = static_cast<Pixel>(colour);
      }
    }

    // One rendering is complete, write data to file.
    fprintf(out, "P3\n"); // PPM Magic number
    fprintf(out, "%lu %lu\n", width, height); // Header line 2
    fprintf(out, "255\n"); // Header line 3: max colour value

    // Iterate over each point in the image, generating and writing
    // pixel data.
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        const Pixel pixel = image[y][x];
        fprintf(out, "%u %u %u ", pixel.r, pixel.g, pixel.b);
      }
      fprintf(out, "\n");
    }
  }

#ifdef DEBUG
  void print() const {
    printf("Scene[%p]:\n", this);

    for (std::vector<Sphere>::const_iterator s = spheres.begin();
         s != spheres.end(); s++) {
      const Sphere sphere = *s;
      sphere.print();
    }

    for (std::vector<Light>::const_iterator l = lights.begin();
         l != lights.end(); l++) {
      const Light light = *l;
      light.print();
    }
  }
#endif
};


#ifdef DEBUG
static const double TEST_ACCURACY = 1e-4;

// Unit tests for colour operations.
void colourTests() {
  printf("Running colour tests...\n");

  Colour a(0, 0xf0, 0xff);
  Colour n = Colour();

  // Default constructor values.
  assert(n.eq(0, 0, 0));

  // Constructor.
  assert(a.eq(0, 240, 255));
}

// Unit tests for vector operations.
void vectorTests() {
  printf("Running vector tests...\n");

  Vector a(0, 1, 2);
  Vector b(0, -1, 1.5);
  Vector c(0, 1, 2);
  Vector n = Vector();

  Vector t;

  // Default constructor values.
  assert(n.eq(0, 0, 0));

  // Constructor.
  assert(a.eq(0, 1, 2));

  // Vector equality testing.
  assert(a == c);
  assert(a != b);

  // Vector scalar multiplication.
  t = a * 2;
  assert(t.eq(0, 2, 4));

  // Vector scalar division.
  t = a / 2;
  assert(t.eq(0, .5, 1));

  // Vector magntidue.
  assert(t.magnitude() - 1.118034 < TEST_ACCURACY);

  // Vector addition.
  t = a + b;
  assert(t.eq(0, 0, 3.5));

  // Vector multiplication.
  t = a * b;
  assert(t.eq(0, -1, 3));

  // Vector normalise.
  t = b.normalise();
  assert(t.x == 0);
  assert(t.y - -0.554700 < TEST_ACCURACY);
  assert(t.z - 0.832050 < TEST_ACCURACY);

  // Dot product.
  assert(dot(a, b) == 2);
}

// Unit tests for ray operations.
void rayTests() {
  printf("Running ray tests...\n");

  Ray a(10, 100);
  Ray n = Ray();

  // Default constructor values.
  assert(n.eq(0, 0, RAY_START_Z, 0, 0, 1));

  // Constructor.
  assert(a.eq(10, 100, RAY_START_Z, 0, 0, 1));
}

// Unit tests for sphere operations.
void sphereTests() {
  printf("Running sphere tests...\n");
}

// Unit tests for light operations.
void lightTests() {
  printf("Running light tests...\n");
}

// Unit tests for scene operations.
void sceneTests() {
  printf("Running scene tests...\n");
}
#endif

// Return the length of array.
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))
// Return the end of an array.
#define ARRAY_END(x) (x + ARRAY_LENGTH(x))

// Program entry point.
int main() {

#ifdef DEBUG
  // Run tests.
  colourTests();
  vectorTests();
  rayTests();
  sphereTests();
#endif

  // The scene:
  const Sphere _spheres[] = {
    Sphere(Vector(125, 250,  300), 75, Material(Colour(0,   200,   5), 1)),
    Sphere(Vector(150, 250,    0), 75, Material(Colour(100,  25,   5), 1)),
    Sphere(Vector(250, 275,  -75), 50, Material(Colour(255, 255, 255), 1)),
    Sphere(Vector(400, 275, -100), 50, Material(Colour(  0, 100, 200), 1))
  };

  const Light _lights[] = {
    Light(Vector(800, -200, -300), Colour(255, 255, 255)),
    Light(Vector(-300, -200, -700), Colour(80, 80, 80))
  };

  // Create the scene to render.
  const std::vector<Sphere> spheres(_spheres, ARRAY_END(_spheres));
  const std::vector<Light> lights(_lights, ARRAY_END(_lights));
  Scene scene(spheres, lights);

  // Output file to write to.
  const char *path = "render.ppm";

  // Open the output file.
  printf("Opening file '%s'...\n", path);
  FILE *const out = fopen(path, "w");

#ifdef DEBUG
  scene.print();
#endif

  // Render the scene to the output file.
  scene.render(512, 512, out);

  // Close the output file.
  printf("Closing file '%s'...\n", path);
  fclose(out);

  return 0;
}
