from unittest import TestCase,skip

import smith
from smith import parboil
from smith import config

@skip("deprecated")
class TestOpenCLDeviceType(TestCase):
    def test_gpu(self):
        self.assertEqual("CL_DEVICE_TYPE_GPU", parboil.OpenCLDeviceType.GPU)

    def test_cpu(self):
        self.assertEqual("CL_DEVICE_TYPE_CPU", parboil.OpenCLDeviceType.CPU)

    def test_to_str_gpu(self):
        d = parboil.OpenCLDeviceType.GPU
        self.assertEqual("GPU", parboil.OpenCLDeviceType.to_str(d))

    def test_to_str_cpu(self):
        d = parboil.OpenCLDeviceType.CPU
        self.assertEqual("CPU", parboil.OpenCLDeviceType.to_str(d))

    def test_to_str_exception(self):
        with self.assertRaises(parboil.InternalException):
            parboil.OpenCLDeviceType.to_str("BAD VALUE")

@skip("deprecated")
class TestScenarioStatus(TestCase):
    def test_good(self):
        self.assertEqual(0, parboil.ScenarioStatus.GOOD)

    def test_bad(self):
        self.assertEqual(1, parboil.ScenarioStatus.BAD)

    def test_unknown(self):
        self.assertEqual(2, parboil.ScenarioStatus.UNKNOWN)

    def test_to_str_good(self):
        g = parboil.ScenarioStatus.GOOD
        self.assertEqual("GOOD", parboil.ScenarioStatus.to_str(g))

    def test_to_str_bad(self):
        b = parboil.ScenarioStatus.BAD
        self.assertEqual("BAD", parboil.ScenarioStatus.to_str(b))

    def test_to_str_unknown(self):
        u = parboil.ScenarioStatus.UNKNOWN
        self.assertEqual("UNKNOWN", parboil.ScenarioStatus.to_str(u))

    def test_to_str_exception(self):
        with self.assertRaises(parboil.InternalException):
            parboil.ScenarioStatus.to_str("BAD VALUE")

@skip("deprecated")
class TestBenchmark(TestCase):
    def test_benchmark(self):
        benchmark = parboil.Benchmark("spmv")

        # scenario = Scenario.get_id()
        self.assertEqual(config.parboil_root(), benchmark.parboil_root)
        self.assertEqual("spmv", benchmark.id)
        self.assertTrue(len(benchmark.datasets))


if __name__ == '__main__':
    main()
