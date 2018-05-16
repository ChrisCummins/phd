from absl import app

from gpu.cldrive.tests.testlib import *


@pytest.mark.skip(reason="FIXME(cec)")
def test_empty_kernel():
  src = " kernel void A() {} "
  outputs = cldrive.drive(ENV, src, [], gsize=(1, 1, 1), lsize=(1, 1, 1))
  assert len(outputs) == 0


@pytest.mark.skip(reason="FIXME(cec)")
def test_simple():
  inputs = [[0, 1, 2, 3, 4, 5, 6, 7]]
  inputs_orig = [[0, 1, 2, 3, 4, 5, 6, 7]]
  outputs_gs = [[0, 2, 4, 6, 8, 10, 12, 14]]

  src = """
    kernel void A(global float* a) {
        const int x_id = get_global_id(0);

        a[x_id] *= 2.0;
    }
    """

  outputs = cldrive.drive(ENV, src, inputs, gsize=(8, 1, 1), lsize=(1, 1, 1))

  almost_equal(inputs, inputs_orig)
  almost_equal(outputs, outputs_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_vector_input():
  inputs = [[0, 1, 2, 3, 0, 1, 2, 3], [2, 4]]
  inputs_orig = [[0, 1, 2, 3, 0, 1, 2, 3], [2, 4]]
  outputs_gs = [[0, 2, 4, 6, 0, 4, 8, 12], [2, 4]]

  src = """
    kernel void A(global int* a, const int2 b) {
        const int x_id = get_global_id(0);
        const int y_id = get_global_id(1);

        if (!y_id) {
            a[x_id] *= b.x;
        } else {
            a[get_global_size(0) + x_id] *= b.y;
        }
    }
    """

  outputs = cldrive.drive(ENV, src, inputs, gsize=(4, 2, 1), lsize=(1, 1, 1))

  almost_equal(inputs, inputs_orig)
  almost_equal(outputs, outputs_gs)

  # run kernel a second time with the previous outputs
  outputs2 = cldrive.drive(ENV, src, outputs, gsize=(4, 2, 1), lsize=(1, 1, 1))
  outputs2_gs = [[0, 4, 8, 12, 0, 16, 32, 48], [2, 4]]
  almost_equal(outputs2, outputs2_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_syntax_error():
  src = "kernel void A(gl ob a l  i nt* a) {}"
  with DevNullRedirect():
    with pytest.raises(cldrive.OpenCLValueError):
      cldrive.drive(ENV, src, [[]], gsize=(1, 1, 1), lsize=(1, 1, 1))


@pytest.mark.skip(reason="FIXME(cec)")
def test_incorrect_num_of_args():
  src = "kernel void A(const int a) {}"
  # too many inputs
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [[1], [2], [3]], gsize=(1, 1, 1), lsize=(1, 1, 1))

  # too few inputs
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [], gsize=(1, 1, 1), lsize=(1, 1, 1))

  # incorrect input width (3 ints instead of one)
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [[1, 2, 3]], gsize=(1, 1, 1), lsize=(1, 1, 1))


@pytest.mark.skip(reason="FIXME(cec)")
def test_timeout():
  # non-terminating kernel
  src = "kernel void A() { while (true) ; }"
  with pytest.raises(cldrive.Timeout):
    cldrive.drive(ENV, src, [], gsize=(1, 1, 1), lsize=(1, 1, 1), timeout=1)


@pytest.mark.skip(reason="FIXME(cec)")
def test_invalid_sizes():
  src = "kernel void A() {}"

  # invalid global size
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [], gsize=(0, -4, 1), lsize=(1, 1, 1))

  # invalid local size
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [], gsize=(1, 1, 1), lsize=(-1, 1, 1))


@pytest.mark.skip(reason="FIXME(cec)")
def test_gsize_smaller_than_lsize():
  src = "kernel void A() {}"
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [], gsize=(4, 1, 1), lsize=(8, 1, 1))


@skip_on_pocl
def test_iterative_increment():
  src = "kernel void A(global int* a) { a[get_global_id(0)] += 1; }"

  d_cl, d_host = [np.arange(16)], np.arange(16)
  for _ in range(8):
    d_host += 1  # perform computation on host
    d_cl = cldrive.drive(ENV, src, d_cl, gsize=(16, 1, 1), lsize=(16, 1, 1))
    almost_equal(d_cl, [d_host])


@pytest.mark.skip(reason="FIXME(cec)")
def test_gsize_smaller_than_data():
  src = "kernel void A(global int* a) { a[get_global_id(0)] = 0; }"

  inputs = [[5, 5, 5, 5, 5, 5, 5, 5]]
  outputs_gs = [[0, 0, 0, 0, 5, 5, 5, 5]]

  outputs = cldrive.drive(ENV, src, inputs, gsize=(4, 1, 1), lsize=(4, 1, 1))

  almost_equal(outputs, outputs_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_zero_size_input():
  src = "kernel void A(global int* a) {}"
  with pytest.raises(ValueError):
    cldrive.drive(ENV, src, [[]], gsize=(1, 1, 1), lsize=(1, 1, 1))


@pytest.mark.skip(reason="FIXME(cec)")
def test_comparison_against_pointer_warning():
  src = """
    kernel void A(global int* a) {
        int id = get_global_id(0);
        if (id < a) a += 1;
    }
    """

  cldrive.drive(ENV, src, [[0]], gsize=(1, 1, 1), lsize=(1, 1, 1))


@pytest.mark.skip(reason="FIXME(cec)")
def test_profiling():
  src = """
    kernel void A(global int* a, constant int* b) {
        const int id = get_global_id(0);
        a[id] *= b[id];
    }
    """

  inputs = [np.arange(16), np.arange(16)]
  outputs_gs = [np.arange(16) ** 2, np.arange(16)]

  with DevNullRedirect():
    outputs = cldrive.drive(
      ENV, src, inputs, gsize=(16, 1, 1), lsize=(16, 1, 1), profiling=True)

  almost_equal(outputs, outputs_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_header():
  src = """
    #include "header.h"
    kernel void A(global DTYPE* a) {
      a[get_global_id(0)] = DOUBLE(a[get_global_id(0)]);
    }
    """

  inputs = [np.arange(16)]
  outputs_gs = [np.arange(16) * 2]

  pp = cldrive.preprocess(src, include_dirs=[data_path("")])
  outputs = cldrive.drive(ENV, pp, inputs, gsize=(16, 1, 1), lsize=(16, 1, 1))
  almost_equal(outputs, outputs_gs)


# TODO: Difftest against cl_launcher from CLSmith for a CLSmith kernel.


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main(
    [cldrive.driver.__file__, __file__, "-v", "--doctest-modules"]))


if __name__ == "__main__":
  app.run(main)
