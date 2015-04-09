#include <algorithm>
#include <atomic>
#include <chrono>
#include <math.h>

#include "tbb/parallel_for.h"

#include "rt.h"

//////////////////////////
// Configurable Options //
//////////////////////////

// The maximum depth to trace rays for.
static const unsigned int MAX_DEPTH = 100;

// The level of precision to use when calculating intersection
// distances.
static const double ROUNDING_ERROR = 1e-6;

// For each pixel at location x,y, we sample N extra points at
// locations normally distributed about x,y. The sample count
// determines the number of extra rays to trace, and the offset
// determines the maximum distance about the origin.
static const size_t ANTIALIASING_SAMPLE_COUNT = 8;
static const double ANTIALIASING_OFFSET = .6;

// Dimensions of rendered image.
static const int IMG_WIDTH = 750;
static const int IMG_HEIGHT = 422;

static const int RENDER_WIDTH = 750;
static const int RENDER_HEIGHT = 422;

// Starting depth of rays.
static const double RAY_START_Z = -1000;

////////////////////
// Implementation //
////////////////////

// A profiling counter that keeps track of how many times we've called
// Renderer::trace().
static std::atomic<long long> traceCounter;

// The random distribution sampler for calculating the offsets of
// stochastic anti-aliasing.
static NormalDistribution sampler(-ANTIALIASING_OFFSET, ANTIALIASING_OFFSET);

NormalDistribution::NormalDistribution(const double min, const double max) {
        std::random_device random;
        generator = std::mt19937(random());
        distribution = std::uniform_real_distribution<double>(min, max);
}

double NormalDistribution::operator()() {
        return distribution(generator);
}

Colour::Colour(const int hex)
                : r(hex >> 16), g((hex >> 8) & 0xff), b(hex & 0xff) {}

Colour::Colour(const float r, const float g, const float b)
                : r(r), g(g), b(b) {}

void Colour::operator+=(const Colour &c) {
        r += c.r;
        g += c.g;
        b += c.b;
}

void Colour::operator/=(const double x) {
        r /= x;
        g /= x;
        b /= x;
}

Colour Colour::operator*(const double x) const {
        return Colour(r * x, g * x, b * x);
}

Colour Colour::operator*(const Colour c) const {
        return Colour(r * (c.r / 255), g * (c.g / 255), b * (c.b / 255));
}

Colour::operator Pixel() const {
        return {clamp(r), clamp(g), clamp(b)};
}

Material::Material(const Colour &colour,
                   const double ambient,
                   const double diffuse,
                   const double specular,
                   const double shininess,
                   const double reflectivity)
                : colour(colour),
                  ambient(ambient),
                  diffuse(diffuse),
                  specular(specular),
                  shininess(shininess),
                  reflectivity(reflectivity) {}

Vector::Vector(const double x, const double y, const double z)
                : x(x), y(y), z(z) {}

Vector Vector::operator+(const Vector &b) const {
        return Vector(x + b.x, y + b.y, z + b.z);
}

Vector Vector::operator-(const Vector &b) const {
        return Vector(x - b.x, y - b.y, z - b.z);
}

Vector Vector::operator*(const double a) const {
        return Vector(a * x, a * y, a * z);
}

Vector Vector::operator/(const double a) const {
        return Vector(x / a, y / a, z / a);
}

Vector Vector::operator*(const Vector &b) const {
        return Vector(x * b.x, y * b.y, z * b.z);
}

bool Vector::operator==(const Vector &b) const {
        return x == b.x && y == b.y && z == b.z;
}

bool Vector::operator!=(const Vector &b) const {
        return !(*this == b);
}

double Vector::size() const {
        return sqrt(x * x + y * y + z * z);
}

double Vector::product() const {
        return x * y * z;
}

double Vector::sum() const {
        return x + y + z;
}

Vector Vector::normalise() const {
        return *this / size();
}

double Vector::operator^(const Vector &b) const {
        return x * b.x + y * b.y + z * b.z;
}

Vector Vector::operator|(const Vector &b) const {
        return Vector(y * b.z - z * b.y,
                      z * b.x - x * b.z,
                      x * b.y - y * b.z);
}

Object::Object(const Vector &position)
                : position(position) {}

Plane::Plane(const Vector &origin,
             const Vector &direction,
             const Material *const material)
                : Object(origin), direction(direction), material(material) {}

Vector Plane::normal(const Vector &p) const {
        return direction;
}

double Plane::intersect(const Ray &ray) const {
        // Calculate intersection of line and plane.
        const double f = (position - ray.position) ^ direction;
        const double g = ray.direction ^ direction;
        const double t = f / g;

        // Accommodate for precision errors.
        const double t0 = t - ROUNDING_ERROR / 2;
        const double t1 = t + ROUNDING_ERROR / 2;

        if (t0 > ROUNDING_ERROR)
                return t0;
        else if (t1 > ROUNDING_ERROR)
                return t1;
        else
                return 0;
}

const Material *Plane::surface(const Vector &point) const {
        return material;
}

// Checkerboard material types.
static const Material CBLACK = Material(Colour(0x888888), 0, .3, 1, 10, 0.7);
static const Material CWHITE = Material(Colour(0x888888), 0, .3, 1, 10, 0.7);

CheckerBoard::CheckerBoard(const Vector &origin,
                           const Vector &direction,
                           const double checkerWidth)
                : Plane(origin, direction, NULL), black(&CBLACK), white(&CWHITE),
                  checkerWidth(checkerWidth) {}

CheckerBoard::~CheckerBoard() {}

const Material *CheckerBoard::surface(const Vector &point) const {
        // TODO: translate point to a relative position on plane.
        const Vector relative = point;

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
               const double radius,
               const Material *const material)
                : Object(position), radius(radius), material(material) {}

Vector Sphere::normal(const Vector &p) const {
        return (p - position).normalise();
}

double Sphere::intersect(const Ray &ray) const {
        // Calculate intersection of line and sphere.
        const Vector distance = position - ray.position;
        const double b = ray.direction ^ distance;
        const double d = b * b + radius * radius - (distance ^ distance);

        if (d < 0)
                return 0;

        const double t0 = b - sqrt(d);
        const double t1 = b + sqrt(d);

        if (t0 > ROUNDING_ERROR)
                return t0;
        else if (t1 > ROUNDING_ERROR)
                return t1;
        else
                return 0;
}

const Material *Sphere::surface(const Vector &point) const {
        return material;
}

PointLight::PointLight(const Vector &position, const Colour &colour)
                : position(position), colour(colour) {};

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

        // Product of material and light colour.
        const Colour illumination = colour * material->colour;

        // Apply Lambert (diffuse) shading.
        const double lambert = std::max(normal ^ toLight,
                                        static_cast<double>(0));
        shade += illumination * material->diffuse * lambert;

        // Apply Blinn-Phong (specular) shading.
        const Vector bisector = (toRay + toLight).normalise();
        const double phong = pow(std::max(normal ^ bisector,
                                          static_cast<double>(0)),
                                 material->shininess);
        shade += illumination * material->specular * phong;

        return shade;
}

Scene::Scene(const std::vector<const Object *> &objects,
             const std::vector<const Light *> &lights)
                : objects(objects), lights(lights) {}

Ray::Ray(const double x, const double y)
                : position(Vector(x, y, RAY_START_Z)),
                  direction(Vector(0, 0, 1)) {}

Ray::Ray(const Vector &position, const Vector &direction)
                : position(position), direction(direction) {}

Renderer::Renderer(const Scene scene)
                : scene(scene), width(RENDER_WIDTH), height(RENDER_HEIGHT) {}

Colour Renderer::supersample(size_t x, size_t y) const {
        Colour sample = Colour();

        // Trace the origin ray.
        sample += trace(Ray(x, y));

        // Accumulate extra samples, normally distributed around x,y.
        for (size_t i = 0; i < ANTIALIASING_SAMPLE_COUNT; i++)
                sample += trace(Ray(x + sampler(), y + sampler()));

        // Average the accumulated samples.
        sample /= ANTIALIASING_SAMPLE_COUNT + 1;

        return sample;
}

void Renderer::render(FILE *const out) const {
        // Image data.
        Pixel image[height * width];

        // For each pixel in the image:
        tbb::parallel_for(
            static_cast<size_t>(0), height * width, [&](size_t i) {
                    const size_t y = i / width;
                    const size_t x = i % width;

                    // Calculate pixel data.
                    image[y * width + x] = supersample(x, y);
            });

        // Once rendering is complete, write data to file.
        fprintf(out, "P3\n"); // PPM Magic number
        fprintf(out, "%lu %lu\n", width, height); // Header line 2
        fprintf(out, "255\n"); // Header line 3: max colour value

        // Iterate over each point in the image, generating and writing
        // pixel data.
        for (size_t y = 0; y < height; y++) {
                for (size_t x = 0; x < width; x++) {
                        const Pixel pixel = image[y * width + x];
                        fprintf(out, "%u %u %u ", pixel.r, pixel.g, pixel.b);
                }
                fprintf(out, "\n");
        }
}

Colour Renderer::trace(const Ray &ray, Colour colour,
                       const unsigned int depth) const {
        // Bump the profiling counter.
        traceCounter++;

        // Determine the closet ray-object intersection.
        double t;
        int index = closestIntersect(ray, scene.objects, t);

        // If the ray doesn't intersect any object, return.
        if (index == -1)
            return colour;

        // Object with closest intersection.
        const Object *object = scene.objects[index];
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
        for (size_t i = 0; i < scene.lights.size(); i++)
                colour += scene.lights[i]->shade(intersect, normal, toRay,
                                                 material, scene.objects);

        // Create reflection ray and recursive evaluate.
        const double reflectivity = material->reflectivity;
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
                     double &t) {
        // Index of, and distance to closest intersect:
        int index = -1;
        t = INFINITY;

        // For each object:
        for (size_t i = 0; i < objects.size(); i++) {
                // Get intersect distance.
                double currentT = objects[i]->intersect(ray);

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

uint8_t inline clamp(const double x) {
        const double min = 0;
        const double max = 255;

        return static_cast<uint8_t>(std::max(std::min(x, max), min));
}

// Return the length of array.
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))
// Return the end of an array.
#define ARRAY_END(x) (x + ARRAY_LENGTH(x))

// Program entry point.
int main() {
        // Material parameters:
        //   colour, ambient, diffuse, specular, shininess, reflectivity
        const Material *const green = new Material(Colour(0x00c805),
                                                   0, 1, .9, 75, 0);
        const Material *const red   = new Material(Colour(0x641905),
                                                   0, 1, .6, 150, 0.4);
        const Material *const mirror = new Material(Colour(0xffffff),
                                                    0, 0, 1, 200, .9999);
        const Material *const grey  = new Material(Colour(0xffffff),
                                                   0, 0.25, 1, 200, .05);
        const Material *const blue  = new Material(Colour(0x0064c8),
                                                   0, .7, .7, 90, 0);

        // The scene:
        const Object *_objects[] = {
                new CheckerBoard(Vector(0, 380, 300),
                                 Vector(0, -30, -1).normalise(), 30),     // Floor
                new Sphere(Vector(150, 240, 300), 135, green),  // Green ball
                new Sphere(Vector(225, 285,   0), 105, red),    // Red ball
                new Sphere(Vector(415, 288, -85), 75,  mirror), // Mirror ball
                new Sphere(Vector(550, 288, -85), 50,  blue),   // Blue ball
                new Sphere(Vector(650, 110,   0), 50,  grey),   // Grey ball
                new Sphere(Vector(650, 210,   0), 50,  grey),   // Grey ball
                new Sphere(Vector(650, 310,   0), 50,  grey)    // Grey ball
        };
        const Light *_lights[] = {
                new PointLight(Vector( 800, 0, -800), Colour(0xffffff)),
                new PointLight(Vector(-300, -200,  -700), Colour(0x105010)),
                new PointLight(Vector( 100, -200,   200), Colour(0x501010))
        };

        // Create the scene and renderer.
        const std::vector<const Object *> objects(_objects, ARRAY_END(_objects));
        const std::vector<const Light *> lights(_lights, ARRAY_END(_lights));
        const Scene scene(objects, lights);
        const Renderer renderer(scene);

        // Output file to write to.
        const char *path = "render.ppm";

        // Open the output file.
        printf("Opening file '%s'...\n", path);
        FILE *const out = fopen(path, "w");

        // Print start message.
        printf("Rendering %d pixels ...\n", RENDER_WIDTH * RENDER_HEIGHT);

        // Record start time.
        const std::chrono::high_resolution_clock::time_point startTime
                        = std::chrono::high_resolution_clock::now();

        // Render the scene to the output file.
        renderer.render(out);

        // Record end time.
        const std::chrono::high_resolution_clock::time_point endTime
                        = std::chrono::high_resolution_clock::now();

        // Close the output file.
        printf("Closing file '%s'...\n\n", path);
        fclose(out);

        // Free heap memory.
        for (size_t i = 0; i < ARRAY_LENGTH(_objects); i++)
                delete _objects[i];
        for (size_t i = 0; i < ARRAY_LENGTH(_lights); i++)
                delete _lights[i];

        // Calculate performance information.
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime).count() / 1e6;
        long long traceCount = static_cast<long long>(traceCounter);
        long long traceRate = traceCount / elapsed;
        long long pixelRate = RENDER_WIDTH * RENDER_HEIGHT / elapsed;
        double tracePerPixel = double(traceCount) / double(RENDER_WIDTH * RENDER_HEIGHT);

        // Print performance summary.
        printf("Rendered %d pixels from %lld traces in %.3f seconds.\n\n",
               RENDER_WIDTH * RENDER_HEIGHT, traceCount, elapsed);
        printf("Render performance:\n");
        printf("\tTraces per second:\t%lld\n", traceRate);
        printf("\tPixels per second:\t%lld\n", pixelRate);
        printf("\tTraces per pixel:\t%.2f\n", tracePerPixel);

        return 0;
}
