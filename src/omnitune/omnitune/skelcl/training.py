import itertools
import random

import labm8 as lab
from labm8 import io

from . import hash_params

WG_VALUES = [4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]


class SampleStrategy(object):
    param_values = WG_VALUES
    unconstrained_space = list(itertools.product(param_values,
                                                 param_values))

    def __init__(self, scenario, max_wg_size, db):
        self.scenario = scenario
        self.db = db

        # Apply max_wg_size constraint to parameter space.
        self.sample_space = [x for x in self.unconstrained_space
                             if x[0] * x[1] < max_wg_size]

        # Generate list of samples.
        self._wgs = []
        self._update_wgs()

    def _update_wgs(self):
        sample_counts = []

        io.debug("Creating sample list...")

        for point in self.sample_space:
            # Get the number of samples at each point in the sample
            # space.
            params = hash_params(*point)
            sample_count = self.db.lookup_runtimes_count(self.scenario, params)
            sample_counts.append((point, sample_count))

        most_samples = max([x[1] for x in sample_counts]) + 5

        jobs = []
        for sample in sample_counts:
            wg = sample[0]
            count = sample[1]
            diff = most_samples - count
            for i in range(diff):
                self._wgs.append(wg)

        random.shuffle(self._wgs)

        possible = len(sample_counts) * 100
        total = sum([x[1] for x in sample_counts])
        self.coverage = float(total) / float(possible)

    def next(self):
        if not len(self._wgs):
            self._update_wgs()
        return self._wgs.pop(0)
