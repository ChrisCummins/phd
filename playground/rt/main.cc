// -*- c-basic-offset: 8; -*-

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <math.h>

#include "tbb/parallel_for.h"

#include "rt.h"

// Generated renderer and image factory:
#include "quick.rt.out"

//////////////////////////
// Configurable Options //
//////////////////////////
#define SEXY 0

// For each pixel at location x,y, we sample N extra points at
// locations randomly distributed about x,y. The sample count
// determines the number of extra rays to trace, and the offset
// determines the maximum distance about the origin.
//
// For softlights, we emit rays at points randomly distributed about
// the light's position. The number of rays emitted is equal to: N =
// (base + radius * factor) ^ 3.
#if SEXY
static const unsigned int MAX_DEPTH           = 100;
static const size_t ANTIALIASING_SAMPLE_COUNT = 8;
static const Scalar ANTIALIASING_OFFSET       = .4;
static const Scalar SOFTLIGHT_FACTOR          = .035;
static const Scalar SOFTLIGHT_BASE            = 3;
static const float RENDER_SCALE               = 40;
#else
static const unsigned int MAX_DEPTH           = 5;
static const size_t ANTIALIASING_SAMPLE_COUNT = 0;
static const Scalar ANTIALIASING_OFFSET       = .6;
static const Scalar SOFTLIGHT_FACTOR          = .01;
static const Scalar SOFTLIGHT_BASE            = 3;
static const float RENDER_SCALE               = 20;
#endif

// Dimensions of "camera" image.
static const int FILM_WIDTH      = 36;
static const int FILM_HEIGHT     = 24;
static const Scalar FOCAL_LENGTH = 30;

// Dimensions of rendered image (output pixels).
static const int RENDER_WIDTH  = FILM_WIDTH * RENDER_SCALE;
static const int RENDER_HEIGHT = FILM_HEIGHT * RENDER_SCALE;

// Gamma of output image.
static const Scalar RENDER_R_GAMMA = 1;
static const Scalar RENDER_G_GAMMA = .98;
static const Scalar RENDER_B_GAMMA = 1.01;

////////////////////
// Implementation //
////////////////////

// A profiling counter that keeps track of how many times we've called
// Renderer::trace().
static std::atomic<long long> traceCounter;

// A profiling counter that keeps track of how many times we've
// contributed light to a ray.
static std::atomic<long long> rayCounter;

// Scene object and light counters.
long long objectsCount;
long long lightsCount;

// The random distribution sampler for calculating the offsets of
// stochastic anti-aliasing.
static UniformDistribution sampler = UniformDistribution(-ANTIALIASING_OFFSET,
                                                         ANTIALIASING_OFFSET);

static const unsigned long long rngMax = 4294967295ULL;
const unsigned long long UniformDistribution::longMax = rngMax;
const Scalar UniformDistribution::scalarMax = rngMax;
const unsigned long long UniformDistribution::mult = 62089911ULL;

UniformDistribution::UniformDistribution(const Scalar min, const Scalar max,
                                         const unsigned long long seed)
                : divisor(scalarMax / (max - min)), min(min), seed(seed) {}

Scalar inline UniformDistribution::operator()() {
        seed *= mult;

        // Generate a new random value in the range [0,max - min].
        const double r = seed % longMax / divisor;
        // Apply "-min" offset to value.
        return r + min;
}

Colour::Colour(const int hex)
                : r((hex >> 16) / 255.),
                  g(((hex >> 8) & 0xff) / 255.),
                  b((hex & 0xff) / 255.) {}

Colour::Colour(const float r, const float g, const float b)
                : r(r), g(g), b(b) {}

void Colour::operator+=(const Colour &c) {
        r += c.r;
        g += c.g;
        b += c.b;
}

void Colour::operator/=(const Scalar x) {
        r /= x;
        g /= x;
        b /= x;
}

Colour Colour::operator*(const Scalar x) const {
        return Colour(r * x, g * x, b * x);
}

Colour Colour::operator/(const Scalar x) const {
        return Colour(r / x, g / x, b / x);
}

Colour Colour::operator*(const Colour &rhs) const {
        return Colour(r * rhs.r, g * rhs.g, b * rhs.b);
}

Colour::operator Pixel() const {
        return {scale(clamp(r)), scale(clamp(g)), scale(clamp(b))};
}

Material::Material(const Colour &colour,
                   const Scalar ambient,
                   const Scalar diffuse,
                   const Scalar specular,
                   const Scalar shininess,
                   const Scalar reflectivity)
                : colour(colour),
                  ambient(ambient),
                  diffuse(diffuse),
                  specular(specular),
                  shininess(shininess),
                  reflectivity(reflectivity) {}

Vector::Vector(const Scalar x, const Scalar y, const Scalar z, const Scalar w)
                : x(x), y(y), z(z), w(w) {}

Vector inline Vector::operator+(const Vector &b) const {
        return Vector(x + b.x, y + b.y, z + b.z);
}

Vector inline Vector::operator-(const Vector &b) const {
        return Vector(x - b.x, y - b.y, z - b.z);
}

Vector inline Vector::operator*(const Scalar a) const {
        return Vector(a * x, a * y, a * z);
}

Vector inline Vector::operator/(const Scalar a) const {
        return Vector(x / a, y / a, z / a);
}

Vector inline Vector::operator*(const Vector &b) const {
        return Vector(x * b.x, y * b.y, z * b.z);
}

bool inline Vector::operator==(const Vector &b) const {
        return x == b.x && y == b.y && z == b.z;
}

bool inline Vector::operator!=(const Vector &b) const {
        return !(*this == b);
}

Scalar inline Vector::size() const {
        return sqrt(x * x + y * y + z * z);
}

Scalar inline Vector::product() const {
        return x * y * z;
}

Scalar inline Vector::sum() const {
        return x + y + z;
}

Vector inline Vector::normalise() const {
        return *this / size();
}

Scalar inline Vector::operator^(const Vector &b) const {
        // Dot product uses the forth component.
        return x * b.x + y * b.y + z * b.z + w * b.w;
}

Vector inline Vector::operator|(const Vector &b) const {
        return Vector(y * b.z - z * b.y,
                      z * b.x - x * b.z,
                      x * b.y - y * b.z);
}

Matrix::Matrix(const Vector r1, const Vector r2,
                               const Vector r3, const Vector r4)
                : r({r1, r2, r3, r4}),
                  c({Vector(r1.x, r2.x, r3.x, r4.x),
                     Vector(r1.y, r2.y, r3.y, r4.y),
                     Vector(r1.z, r2.z, r3.z, r4.z),
                     Vector(r1.w, r2.w, r3.w, r4.w)}) {}

Matrix Matrix::operator*(const Matrix &b) const {
        return Matrix(
            Vector(r[0] ^ b.c[0], r[0] ^ b.c[1], r[0] ^ b.c[2], r[0] ^ b.c[3]),
            Vector(r[1] ^ b.c[0], r[1] ^ b.c[1], r[1] ^ b.c[2], r[1] ^ b.c[3]),
            Vector(r[2] ^ b.c[0], r[2] ^ b.c[1], r[2] ^ b.c[2], r[2] ^ b.c[3]),
            Vector(r[3] ^ b.c[0], r[3] ^ b.c[1], r[3] ^ b.c[2], r[3] ^ b.c[3]));
}

Vector Matrix::operator*(const Vector &b) const {
        // Pad the "w" component.
        const Vector v = Vector(b.x, b.y, b.z, 1);
        return Vector(r[0] ^ v, r[1] ^ v, r[2] ^ v, r[3] ^ v);
}

Matrix Matrix::operator*(const Scalar a) const {
        return Matrix(r[0] * a, r[1] * a, r[2] * a, r[3] * a);
}

Translation::Translation(const Scalar x, const Scalar y, const Scalar z)
                : Matrix(Vector(1, 0, 0, x),
                         Vector(0, 1, 0, y),
                         Vector(0, 0, 1, z),
                         Vector(0, 0, 0, 1)) {}

Translation::Translation(const Vector &t)
                : Matrix(Vector(1, 0, 0, t.x),
                         Vector(0, 1, 0, t.y),
                         Vector(0, 0, 1, t.z),
                         Vector(0, 0, 0, 1)) {}

Scale::Scale(const Scalar x, const Scalar y, const Scalar z)
                : Matrix(Vector(x, 0, 0, 0),
                         Vector(0, y, 0, 0),
                         Vector(0, 0, z, 0),
                         Vector(0, 0, 0, 1)) {}

Scale::Scale(const Vector &s)
                : Matrix(Vector(s.x, 0, 0, 0),
                         Vector(0, s.y, 0, 0),
                         Vector(0, 0, s.z, 0),
                         Vector(0, 0, 0, 1)) {}

Matrix rotation(const Scalar x, const Scalar y, const Scalar z) {
        return RotationZ(z) * RotationY(y) * RotationX(x);
}

RotationX::RotationX(const Scalar theta)
                : Matrix(Vector(1, 0, 0, 0),
                         Vector(0, dcos(theta), -dsin(theta), 0),
                         Vector(0, dsin(theta), dcos(theta), 0),
                         Vector(0, 0, 0, 1)) {}

RotationY::RotationY(const Scalar theta)
                : Matrix(Vector(dcos(theta), 0, dsin(theta), 0),
                         Vector(0, 1, 0, 0),
                         Vector(-dsin(theta), 0, dcos(theta), 0),
                         Vector(0, 0, 0, 1)) {}

RotationZ::RotationZ(const Scalar theta)
                : Matrix(Vector(dcos(theta), -dsin(theta), 0, 0),
                         Vector(dsin(theta), dcos(theta), 0, 0),
                         Vector(0, 0, 1, 0),
                         Vector(0, 0, 0, 1)) {}

Scalar ONB::epsilon = 0.01;

bool inline ONB::operator==(const ONB &rhs) {
        return u == rhs.u && v == rhs.v && w == rhs.w;
}

bool inline ONB::operator!=(const ONB &rhs) {
        return !(*this == rhs);
}

ONB ONB::initFromU(const Vector &u) {
        const Vector n(1, 0, 0);
        const Vector m(0, 1, 0);

        const Vector u1 = u.normalise();
        const Vector u2 = u1 | n;
        const Vector u3 = u1 | m;
        const Vector v = u2.size() < ONB::epsilon ? u3 : u2;
        const Vector w = u1 | v;

        return ONB(u, v, w);
}

ONB ONB::initFromV(const Vector &v) {
        const Vector n(1, 0, 0);
        const Vector m(0, 1, 0);

        const Vector v1 = v.normalise();
        const Vector u0 = v1 | n;
        const Vector u1 = v1 | m;
        const Vector u = u0.size() * u0.size() < epsilon ? u1 : u0;
        const Vector w = u | v1;

        return ONB(u, v, w);
}

ONB ONB::initFromW(const Vector &w) {
        const Vector n(1, 0, 0);
        const Vector m(0, 1, 0);

        const Vector w1 = w.normalise();
        const Vector u0 = w1 | n;
        const Vector u1 = w1 | m;
        const Vector u = u0.size() < epsilon ? u1 : u0;
        const Vector v = w1 | u;

        return ONB(u, v, w);
}

ONB ONB::initFromUV(const Vector &u, const Vector &v) {
        const Vector u1 = u.normalise();
        const Vector w = (u1 | v).normalise();
        const Vector v1 = w | u1;

        return ONB(u1, v1, w);
}

ONB ONB::initFromVU(const Vector &v, const Vector &u) {
        const Vector v1 = v.normalise();
        const Vector w = (u | v1).normalise();
        const Vector u1 = v1 | w;

        return ONB(u1, v1, w);
}

ONB ONB::initFromUW(const Vector &u, const Vector &w) {
        const Vector u1 = u.normalise();
        const Vector v = w | u1;
        const Vector w1 = u1 | v;

        return ONB(u1, v, w1);
}

ONB ONB::initFromWU(const Vector &w, const Vector &u) {
        const Vector w1 = w.normalise();
        const Vector v = (w1 | u).normalise();
        const Vector u1 = v | w1;

        return ONB(u1, v, w1);
}

ONB ONB::initFromVW(const Vector &v, const Vector &w) {
        const Vector v1 = v.normalise();
        const Vector u = (v1 | w).normalise();
        const Vector w1 = u | v1;

        return ONB(u, v1, w1);
}

ONB ONB::initFromWV(const Vector &w, const Vector &v) {
        const Vector w1 = w.normalise();
        const Vector u = v | w1;
        const Vector v1 = w1 | u;

        return ONB(u, v1, w1);
}

Scalar inline dsin(const Scalar theta) {
        return sin(theta * M_PI / 180.0);
}

Scalar inline dcos(const Scalar theta) {
        return cos(theta * M_PI / 180.0);
}

Scalar inline datan(const Scalar theta) {
        return atan(theta) * 180.0 / M_PI;
}

Object::Object(const Vector &position)
                : position(position) {
        // Register object with profiling counter.
        objectsCount += 1;
}

Plane::Plane(const Vector &origin,
             const Vector &direction,
             const Material *const material)
                : Object(origin), direction(direction), material(material) {}

Vector Plane::normal(const Vector &p) const {
        return direction;
}

Scalar Plane::intersect(const Ray &ray) const {
        // Calculate intersection of line and plane.
        const Scalar f = (position - ray.position) ^ direction;
        const Scalar g = ray.direction ^ direction;
        const Scalar t = f / g;

        // Accommodate for precision errors.
        const Scalar t0 = t - ScalarPrecision / 2;
        const Scalar t1 = t + ScalarPrecision / 2;

        if (t0 > ScalarPrecision)
                return t0;
        else if (t1 > ScalarPrecision)
                return t1;
        else
                return 0;
}

const Material *Plane::surface(const Vector &point) const {
        return material;
}

// Checkerboard material types.
static const Material CBLACK = Material(Colour(0x505050), 0, .07, 1, 10, 0.999);
static const Material CWHITE = Material(Colour(0xffffff), 0, .07, 1, 10, 0.999);

CheckerBoard::CheckerBoard(const Vector &origin,
                           const Vector &direction,
                           const Scalar checkerWidth)
                : Plane(origin, direction, NULL), black(&CBLACK), white(&CWHITE),
                  checkerWidth(checkerWidth) {}

CheckerBoard::~CheckerBoard() {}

static const Scalar gridOffset = 3000000;

const Material *CheckerBoard::surface(const Vector &point) const {
        // TODO: translate point to a relative position on plane.
        const Vector relative = Vector(point.x + gridOffset,
                                       point.z + gridOffset, 0);

        const int x = relative.x;
        const int y = relative.y;
        const int half = static_cast<int>(checkerWidth * 2);
        const int mod = half * 2;

        if (x % mod < half)
                return y % mod < half ? black : white;
        else
                return y % mod < half ? white : black;
}

Sphere::Sphere(const Vector &position,
               const Scalar radius,
               const Material *const material)
                : Object(position), radius(radius), material(material) {}

Vector Sphere::normal(const Vector &p) const {
        return (p - position).normalise();
}

Scalar Sphere::intersect(const Ray &ray) const {
        // Calculate intersection of line and sphere.
        const Vector distance = position - ray.position;
        const Scalar b = ray.direction ^ distance;
        const Scalar d = b * b + radius * radius - (distance ^ distance);

        if (d < 0)
                return 0;

        const Scalar t0 = b - sqrt(d);
        const Scalar t1 = b + sqrt(d);

        if (t0 > ScalarPrecision)
                return t0;
        else if (t1 > ScalarPrecision)
                return t1;
        else
                return 0;
}

const Material *Sphere::surface(const Vector &point) const {
        return material;
}

PointLight::PointLight(const Vector &position, const Colour &colour)
                : position(position), colour(colour) {
        // Register light with profiling counter.
        lightsCount += 1;
};

Colour PointLight::shade(const Vector &point,
                         const Vector &normal,
                         const Vector &toRay,
                         const Material *const material,
                         const std::vector<const Object *> objects) const {
        // Shading is additive, starting with black.
        Colour shade = Colour();

        // Direction vector from point to light.
        const Vector toLight = (position - point).normalise();
        // Determine with light is blocked.
        const bool blocked = intersects(Ray(point, toLight), objects);
        // Do nothing without line of sight.
        if (blocked)
                return shade;

        // Bump the profiling counter.
        rayCounter++;

        // Product of material and light colour.
        const Colour illumination = colour * material->colour;

        // Apply Lambert (diffuse) shading.
        const Scalar lambert = std::max(normal ^ toLight,
                                        static_cast<Scalar>(0));
        shade += illumination * material->diffuse * lambert;

        // Apply Blinn-Phong (specular) shading.
        const Vector bisector = (toRay + toLight).normalise();
        const Scalar phong = pow(std::max(normal ^ bisector,
                                          static_cast<Scalar>(0)),
                                 material->shininess);
        shade += illumination * material->specular * phong;

        return shade;
}

static UniformDistribution softSampler = UniformDistribution(-1, 1);

SoftLight::SoftLight(const Vector &position, const Scalar radius,
                     const Colour &colour)
                : position(position), radius(radius), colour(colour),
                  samples(SOFTLIGHT_BASE +
                          std::pow(radius * SOFTLIGHT_FACTOR, 3)) {
        // Register lights with profiling counter.
        lightsCount += samples;
};

Colour SoftLight::shade(const Vector &point,
                        const Vector &normal,
                        const Vector &toRay,
                        const Material *const material,
                        const std::vector<const Object *> objects) const {
        // Shading is additive, starting with black.
        Colour shade = Colour();

        // Product of material and light colour.
        const Colour illumination = (colour * material->colour) / samples;

        // Cast multiple light rays, nomrally distributed about the
        // light's centre.
        for (size_t i = 0; i < samples; i++) {
                const Vector origin = Vector(position.x + softSampler() * radius,
                                             position.y + softSampler() * radius,
                                             position.z + softSampler() * radius);

                // Direction vector from point to light.
                const Vector toLight = (origin - point).normalise();
                // Determine whether the light is blocked.
                const bool blocked = intersects(Ray(point, toLight), objects);
                // Do nothing without line of sight.
                if (blocked)
                        continue;

                // Bump the profiling counter.
                rayCounter++;

                // Apply Lambert (diffuse) shading.
                const Scalar lambert = std::max(normal ^ toLight,
                                                static_cast<Scalar>(0));
                shade += illumination * material->diffuse * lambert;

                // Apply Blinn-Phong (specular) shading.
                const Vector bisector = (toRay + toLight).normalise();
                const Scalar phong = pow(std::max(normal ^ bisector,
                                                  static_cast<Scalar>(0)),
                                         material->shininess);
                shade += illumination * material->specular * phong;
        }

        return shade;
}

Scene::Scene(const std::vector<const Object *> &objects,
             const std::vector<const Light *> &lights)
                : objects(objects), lights(lights) {}

Scene::~Scene() {
        for (size_t i = 0; i < objects.size(); i++)
                delete objects[i];
        for (size_t i = 0; i < lights.size(); i++)
                delete lights[i];
}

Ray::Ray(const Vector &position, const Vector &direction)
                : position(position), direction(direction) {}

Camera::Camera(const Vector &position,
               const Vector &lookAt,
               const Vector &up,
               const Scalar width,
               const Scalar height,
               const Scalar focalLength)
                : position(position),
                  direction((lookAt - position).normalise()),
                  filmBack(position - (lookAt - position).normalise()
                           * focalLength),
                  right((lookAt - position).normalise() | up),
                  up(((lookAt - position).normalise() | up) |
                     (lookAt - position).normalise()),
                  width(width),
                  height(height) {}

Image::Image(const size_t width, const size_t height,
             const Colour gamma, const bool inverted)
                : image(new Pixel[width * height]),
                  width(width), height(height),
                  power(Colour(1 / gamma.r, 1 / gamma.g, 1 / gamma.b)),
                  inverted(inverted) {}

Image::~Image() {
        // Free pixel data.
        delete[] image;
}

void inline Image::set(const size_t x, const size_t y,
                       const Colour &value) const {
        // Apply Y axis inversion if needed.
        const size_t row = inverted ? height - 1 - y : y;

        // Apply gamma correction.
        const Colour corrected = Colour(std::pow(value.r, power.r),
                                        std::pow(value.g, power.g),
                                        std::pow(value.b, power.b));

        // Explicitly cast colour to pixel data.
        image[row * width + x] = static_cast<Pixel>(corrected);
}

void Image::write(FILE *const out) const {
        // Print PPM header.
        fprintf(out, "P3\n"); // Magic number
        fprintf(out, "%lu %lu\n", width, height); // Image dimensions
        fprintf(out, "%d\n", PixelColourMax); // Max colour value

        // Iterate over each point in the image, writing pixel data.
        for (size_t i = 0; i < height * width; i++) {
                const Pixel pixel = image[i];
                fprintf(out,
                        PixelFormatString" "
                        PixelFormatString" "
                        PixelFormatString" ",
                        pixel.r, pixel.g, pixel.b);

                if (!i % width) // Add newline at the end of each row.
                        fprintf(out, "\n");
        }
}

Renderer::Renderer(const Scene *const scene,
                   const Camera *const camera)
                : scene(scene), camera(camera) {}

Renderer::~Renderer() {
        delete scene;
        delete camera;
}

Colour Renderer::supersample(const Ray &ray) const {
        Colour sample = Colour();

        // Trace the origin ray.
        sample += trace(ray);

// For fast builds, we disable antialiasing. This sets
// ANTIALIASING_SAMPLE_COUNT to 0, which causes the compiler to kick
// up a fuss about comparing 0 < 0. Let's disable that warning for
// such builds.
#if !SEXY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
        // Accumulate extra samples, randomly distributed around x,y.
        for (size_t i = 0; i < ANTIALIASING_SAMPLE_COUNT; i++) {
                const Vector origin = Vector(ray.position.x + sampler(),
                                             ray.position.y + sampler(),
                                             ray.position.z);
                sample += trace(Ray(origin, ray.direction));
        }
#if !SEXY
#pragma GCC diagnostic pop
#endif

        // Average the accumulated samples.
        sample /= ANTIALIASING_SAMPLE_COUNT + 1;

        return sample;
}

void Renderer::render(const Image *const image) const {
        // Scale image coordinates to camera coordinates.
        const Scale scale(camera->width / image->width,
                          camera->height / image->height, 1);
        // Offset from image coordinates to camera coordinates.
        const Translation offset(-(image->width * .5),
                                 -(image->height * .5), 0);
        // Create combined transformation matrix.
        const Matrix transform = scale * offset;

        // For each pixel in the image:
        tbb::parallel_for(
            static_cast<size_t>(0), image->height * image->width, [&](size_t i) {
                    // Image space coordinates.
                    const Scalar x = i % image->width;
                    const Scalar y = i / image->width;

                    // Convert image to camera (local) space coordinates.
                    const Vector localPosition = transform * Vector(x, y, 0);

                    // Translate camera (local) space to world space.
                    const Vector lensPoint =
                                    camera->right * localPosition.x +
                                    camera->up * localPosition.y +
                                    camera->position;

                    // Determine direction from point on lens to exposure point.
                    const Vector direction =
                                    (lensPoint - camera->filmBack).normalise();

                    // Create a ray.
                    const Ray ray = Ray(camera->filmBack, direction);

                    // Sample the ray.
                    image->set(x, y, supersample(ray));
            });
}

Colour Renderer::trace(const Ray &ray, Colour colour,
                       const unsigned int depth) const {
        // Bump the profiling counter.
        traceCounter++;

        // Determine the closet ray-object intersection.
        Scalar t;
        int index = closestIntersect(ray, scene->objects, t);

        // If the ray doesn't intersect any object, return.
        if (index == -1)
            return colour;

        // Object with closest intersection.
        const Object *object = scene->objects[index];
        // Point of intersection.
        const Vector intersect = ray.position + ray.direction * t;
        // Surface normal at point of intersection.
        const Vector normal = object->normal(intersect);
        // Direction between intersection and source ray.
        const Vector toRay = (ray.position - intersect).normalise();
        // Material at point of intersection.
        const Material *material = object->surface(intersect);

        // Apply ambient lighting.
        colour += material->colour * material->ambient;

        // Apply shading from each light source.
        for (size_t i = 0; i < scene->lights.size(); i++)
                colour += scene->lights[i]->shade(intersect, normal, toRay,
                                                 material, scene->objects);

        // Create reflection ray and recursive evaluate.
        const Scalar reflectivity = material->reflectivity;
        if (depth < MAX_DEPTH && reflectivity > 0) {
                // Direction of reflected ray.
                const Vector reflectionDirection = (normal * 2*(normal ^ toRay) - toRay).normalise();
                // Create a reflection.
                const Ray reflection(intersect, reflectionDirection);
                // Add reflection light.
                colour += trace(reflection, colour, depth + 1) * reflectivity;
        }

        return colour;
}

int closestIntersect(const Ray &ray,
                     const std::vector<const Object *> &objects,
                     Scalar &t) {
        // Index of, and distance to closest intersect:
        int index = -1;
        t = INFINITY;

        // For each object:
        for (size_t i = 0; i < objects.size(); i++) {
                // Get intersect distance.
                Scalar currentT = objects[i]->intersect(ray);

                // Check if intersects, and if so, whether the
                // intersection is closer than the current best.
                if (currentT != 0 && currentT < t) {
                        // New closest intersection.
                        t = currentT;
                        index = static_cast<int>(i);
                }
        }

        return index;
}

bool intersects(const Ray &ray, const std::vector<const Object *> &objects) {
        // Iterate over all objects:
        for (size_t i = 0; i < objects.size(); i++)
                // If the ray intersects object, return true.
                if (objects[i]->intersect(ray) != 0)
                        return true;

        // No intersect.
        return false;
}

Scalar inline clamp(const Scalar x) {
        if (x > 1)
                return 1;
        if (x < 0)
                return 0;
        else
                return x;
}

PixelColourType inline scale(const Scalar x) {
        // Scale value.
        const Scalar scaled = x * static_cast<Scalar>(PixelColourMax);

        // Cast to output type.
        return static_cast<PixelColourType>(scaled);
}

// Return the length of array.
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))
// Return the end of an array.
#define ARRAY_END(x) (x + ARRAY_LENGTH(x))

// Program entry point.
int main() {
        // Get the renderer and image.
        const Renderer *const renderer = getRenderer();
        const Image *const image = getImage();

        // Print start message.
        printf("Rendering %d pixels with %lu samples per pixel, "
               "%lld objects, and %lld light sources ...\n",
               RENDER_WIDTH * RENDER_HEIGHT,
               1 + ANTIALIASING_SAMPLE_COUNT,
               objectsCount, lightsCount);

        // Record start time.
        const std::chrono::high_resolution_clock::time_point startTime
                        = std::chrono::high_resolution_clock::now();

        // Render the scene to the output file.
        renderer->render(image);

        // Record end time.
        const std::chrono::high_resolution_clock::time_point endTime
                        = std::chrono::high_resolution_clock::now();

        // Open the output file.
        const char *path = "render.ppm";
        printf("Opening file '%s'...\n", path);
        FILE *const out = fopen(path, "w");

        // Write to output file.
        image->write(out);

        // Close the output file.
        printf("Closing file '%s'...\n\n", path);
        fclose(out);

        // Free heap memory.
        delete renderer;
        delete image;

        // Calculate performance information.
        Scalar elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime).count() / 1e6;
        long long traceCount = static_cast<long long>(traceCounter);
        long long rayCount = static_cast<long long>(rayCounter);
        long long traceRate = traceCount / elapsed;
        long long rayRate = rayCount / elapsed;
        long long pixelRate = RENDER_WIDTH * RENDER_HEIGHT / elapsed;
        Scalar tracePerPixel = static_cast<Scalar>(traceCount)
            / static_cast<Scalar>(RENDER_WIDTH * RENDER_HEIGHT);

        // Print performance summary.
        printf("Rendered %d pixels from %lld traces in %.3f seconds.\n\n",
               RENDER_WIDTH * RENDER_HEIGHT, traceCount, elapsed);
        printf("Render performance:\n");
        printf("\tRays per second:\t%lld\n", rayRate);
        printf("\tTraces per second:\t%lld\n", traceRate);
        printf("\tPixels per second:\t%lld\n", pixelRate);
        printf("\tTraces per pixel:\t%.2f\n", tracePerPixel);

        return 0;
}
