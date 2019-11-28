#include "./trisycl.h"

#include "./003-parallel-matrix.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

matrix read_ppm(const std::string& path) {
  auto numrows = 0u, numcols = 0u;
  std::ifstream in_file(path);

  // function to get next line which is not a comment
  auto next_line = [](std::ifstream& file) -> std::string {
    std::string line;
    while (true) {
      getline(file, line);
      if (line[0] != '#') break;
    }
    return line;
  };

  auto version = next_line(in_file);
  auto line = next_line(in_file);
  std::stringstream(line) >> numcols >> numrows;
  next_line(in_file);  // skip the next line (max value)

  matrix inputImage(numrows, numcols);
  auto iter = inputImage.begin();
  if (version == "P2") {
    while (getline(in_file, line)) {
      std::stringstream(line) >> *iter;
      iter++;
    }
  } else {
    throw std::invalid_argument("read_ppm(): file not P2 type");
  }

  return inputImage;
}

void write_ppm(const matrix& img, const std::string& filename) {
  std::ofstream outputFile(filename);
  outputFile << "P2\n"
             << "#Creator: skelcl\n"
             << img.ncols() << " " << img.ncols() << "\n255\n";

  for (auto& pixel : img) {
    outputFile << static_cast<int>(pixel) << "\n";
  }
}

//
// Sequential Gaussian blur algorithm
//
matrix gaussian_blur(const matrix& input, const std::vector<float>& weights,
                     const size_t radius) {
  matrix output(input.nrows(), input.ncols());

  for (int y = 0; y < static_cast<int>(input.nrows()); y++) {
    for (int x = 0; x < static_cast<int>(input.ncols()); x++) {
      float sum = 0;
      float norm = 0;
      for (int j = -static_cast<int>(radius); j <= static_cast<int>(radius);
           j++) {
        for (int i = -static_cast<int>(radius); i <= static_cast<int>(radius);
             i++) {
          int ly = y + j;
          int lx = x + i;

          if (ly < 0)
            ly = 0;
          else if (ly >= static_cast<int>(input.nrows()))
            ly = static_cast<int>(input.nrows() - 1);

          if (lx < 0)
            lx = 0;
          else if (lx >= static_cast<int>(input.ncols()))
            lx = static_cast<int>(input.ncols() - 1);

          auto wi = static_cast<size_t>((j + i) / 2 + static_cast<int>(radius));

          sum += input.at(static_cast<size_t>(ly), static_cast<size_t>(lx)) *
                 weights[wi];
          norm += weights[wi];
        }
      }
      float v = sum / norm;
      output.at(static_cast<size_t>(y), static_cast<size_t>(x)) =
          (v > 255) ? 255 : ((v < 0) ? 0 : v);
    }
  }

  return output;
}

//
// SYCL Gaussian blur
//
matrix sycl_gaussian_blur(const matrix& input,
                          const std::vector<float>& weights,
                          const size_t radius) {
  matrix output(input.nrows(), input.ncols());

  {
    cl::sycl::queue myQueue;
    cl::sycl::buffer<float, 2> dev_img(
        input.data(), cl::sycl::range<2>{input.nrows(), input.ncols()});
    cl::sycl::buffer<float> dev_weights(weights.data(),
                                        cl::sycl::range<1>{weights.size()});
    cl::sycl::buffer<size_t> dev_radius(&radius, cl::sycl::range<1>{1});
    cl::sycl::buffer<float, 2> dev_out(
        output.data(), cl::sycl::range<2>{output.nrows(), output.ncols()});

    myQueue.submit([&](cl::sycl::handler& cgh) {
      auto kimg = dev_img.get_access<cl::sycl::access::read>(cgh);
      auto kweights = dev_weights.get_access<cl::sycl::access::read>(cgh);
      auto kradius = dev_radius.get_access<cl::sycl::access::read>(cgh);
      auto kout = dev_out.get_access<cl::sycl::access::write>(cgh);
      cgh.parallel_for(
          cl::sycl::range<2>{output.nrows(), output.ncols()},
          [=](const cl::sycl::id<2> id) {
            const int y = static_cast<int>(id.get(0));
            const int x = static_cast<int>(id.get(1));

            float sum = 0;
            float norm = 0;

            for (int j = -static_cast<int>(kradius[0]);
                 j <= static_cast<int>(kradius[0]); j++) {
              for (int i = -static_cast<int>(kradius[0]);
                   i <= static_cast<int>(kradius[0]); i++) {
                int ly = y + j;
                int lx = x + i;

                //
                // TODO: We need to get the dimensions of the global
                // space, rather than using input.nrows() /
                // input.ncols().
                //
                if (ly < 0)
                  ly = 0;
                else if (ly >= static_cast<int>(input.nrows()))
                  ly = static_cast<int>(input.nrows()) - 1;

                if (lx < 0)
                  lx = 0;
                else if (lx >= static_cast<int>(input.ncols()))
                  lx = static_cast<int>(input.ncols()) - 1;

                auto wi = static_cast<size_t>((j + i) / 2 +
                                              static_cast<int>(kradius[0]));

                const cl::sycl::id<2> imgi{static_cast<size_t>(ly),
                                           static_cast<size_t>(lx)};
                sum += kimg[imgi] * kweights[wi];
                norm += kweights[wi];
              }
            }

            float v = sum / norm;
            kout[id] = (v > 255) ? 255 : ((v < 0) ? 0 : v);
          });
    });
  }

  return output;
}

std::vector<float> get_weights(const size_t& radius) {
  const int fwhm = 5;
  const size_t diameter = 2 * radius + 1;
  const float a = (fwhm / 2.354f);

  std::vector<float> weights(diameter);
  for (int i = -static_cast<int>(radius); i <= static_cast<int>(radius); i++)
    weights[static_cast<size_t>(i + static_cast<int>(radius))] =
        exp(-i * i / (2 * a * a));

  return weights;
}

template <matrix Gaussian(const matrix& input,
                          const std::vector<float>& weights,
                          const size_t radius)>
void test_main(const std::string& in_path, const std::string& out_path) {
  const size_t radius = 5;
  auto weights = get_weights(radius);
  auto in_image = read_ppm(in_path);
  std::cout << "Read '" << in_path << "' ...\n";
  auto out_image = Gaussian(in_image, weights, radius);
  write_ppm(out_image, out_path);
  std::cout << "Wrote '" << out_path << "'\n";
}

///////////
// Tests //
///////////

TEST(GaussianBlur, gaussian_blur) {
  test_main<gaussian_blur>("in.pgm", "out.pgm");
}

TEST(GaussianBlur, sycl_gaussian_blur) {
  test_main<sycl_gaussian_blur>("in.pgm", "out-sycl.pgm");
}

PHD_MAIN();
