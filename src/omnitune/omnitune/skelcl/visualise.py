#!/usr/bin/env python2

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from . import space as _space

import labm8 as lab
from labm8 import io
from labm8 import fs
from labm8 import math as labmath
from labm8 import viz


def max_wgsizes(db, output=None, trisurf=False, **kwargs):
    space = db.max_wgsize_space()
    if "title" not in kwargs: kwargs["title"] = ("Distribution of maximum "
                                                 "workgroup sizes")
    if trisurf:
        space.trisurf(output=output, **kwargs)
    else:
        space.heatmap(output=output, **kwargs)


def coverage(db, output=None, where=None, **kwargs):
    space = db.param_coverage_space(where=where)
    if "title" not in kwargs: kwargs["title"] = "All data"
    if "vim" not in kwargs: kwargs["vmin"] = 0
    if "vmax" not in kwargs: kwargs["vmax"] = 1
    space.heatmap(output=output, **kwargs)


def safety(db, output=None, where=None, **kwargs):
    space = db.param_safe_space(where=where)
    if "title" not in kwargs: kwargs["title"] = "All data"
    if "vim" not in kwargs: kwargs["vmin"] = 0
    if "vmax" not in kwargs: kwargs["vmax"] = 1
    space.heatmap(output=output, **kwargs)


def oracle_wgsizes(db, output=None, where=None, trisurf=False, **kwargs):
    space = db.oracle_param_space(where=where)
    if "title" not in kwargs: kwargs["title"] = "All data"
    if trisurf:
        space.trisurf(output=output, **kwargs)
    else:
        if "vim" not in kwargs: kwargs["vmin"] = 0
        if "vmax" not in kwargs: kwargs["vmax"] = 1
        space.heatmap(output=output, **kwargs)


def scenario_performance(db, scenario, output=None, title=None):
    space = _space.ParamSpace.from_dict(db.perf_scenario(scenario))
    space.heatmap(output=output, title=title)


def performance_vs_coverage(db, output=None, figsize=None,
                            title="Workgroup size performance vs. legality"):
    data = sorted([
        (
            db.perf_param_avg(param) * 100,
            db.perf_param_avg_legal(param) * 100,
            db.param_coverage(param) * 100
        )
        for param in db.params
        if db.perf_param_avg_legal(param) > 0
    ], reverse=True, key=lambda x: (x[0], x[2], x[1]))
    X = np.arange(len(data))

    GeoPerformance, Performance, Coverage = zip(*data)

    ax = plt.subplot(111)
    ax.plot(X, Coverage, 'r', label="Legality")
    ax.plot(X, Performance, 'g', label="Performance (when legal)")
    ax.plot(X, GeoPerformance, 'b', label="Performance")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.title(title)
    plt.ylabel("Performance / Legality")
    plt.xlabel("Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    if figsize is not None:
        plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)


def num_params_vs_accuracy(db, output=None, where=None, figsize=None,
                           title=("Number of workgroup sizes "
                                  "vs. oracle accuracy")):
    freqs = sorted(db.oracle_param_frequencies(normalise=True).values(),
                   reverse=True)
    acc = 0
    Data = [0] * len(freqs)
    for i,freq in enumerate(freqs):
        acc += freq * 100
        Data[i] = acc

    X = np.arange(len(Data))
    ax = plt.subplot(111)
    ax.plot(X, Data)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of distinct workgroup sizes")
    plt.tight_layout()
    plt.legend(frameon=True)
    if figsize is not None:
        plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)


def performance_vs_max_wgsize(db, output=None, figsize=None,
                              title=("Workgroup size performance "
                                     "vs. maximum workgroup size")):
    data = sorted([
        (
            db.perf(scenario, param) * 100,
            db.ratio_max_wgsize(scenario, param) * 100
        )
        for scenario, param in db.scenario_params
    ], reverse=True)
    X = np.arange(len(data))

    Performance, Ratios = zip(*data)

    ax = plt.subplot(111)
    ax.plot(X, Ratios, 'g', label="Ratio max wgsize")
    ax.plot(X, Performance, 'b', label="Performance")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.title(title)
    plt.ylabel("Performance / Size")
    plt.xlabel("Scenarios, Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    if figsize is not None:
        plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)


def _performance_plot(output, labels, values, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(values)
    ax.set_xticklabels(labels, rotation=90)
    plt.ylim(ymin=0, ymax=1)
    if "title" in kwargs:
        plt.title(kwargs["title"])
    if "figsize" in kwargs:
        plt.gcf().set_size_inches(*kwargs["figsize"], dpi=300)
    viz.finalise(output)


def kernel_performance(db, output=None, **kwargs):
    labels = db.kernel_names
    values = [db.performance_of_kernels_with_name(label) for label in labels]
    if "title" not in kwargs:
        kwargs["title"] = "Workgroup size performance across kernels"
    _performance_plot(output, labels, values, **kwargs)


def device_performance(db, output=None, **kwargs):
    labels = db.cpus + db.gpus # Arrange CPUs on the left, GPUs on the right.
    values = [db.performance_of_device(label) for label in labels]
    if "title" not in kwargs:
        kwargs["title"] = "Workgroup size performance across devices"
    _performance_plot(output, labels, values, **kwargs)


def dataset_performance(db, output=None, **kwargs):
    labels = db.datasets
    values = [db.performance_of_dataset(label) for label in labels]
    if "title" not in kwargs:
        kwargs["title"] = "Workgroup size performance across datasets"
    _performance_plot(output, labels, values, **kwargs)


def sample_counts(db, output=None, where=None, nbins=25,
                  title="Sample counts for unique scenarios and params"):
    data = sorted([t[2] for t in db.num_samples(where=where)])
    bins = np.linspace(min(data), max(data), nbins)
    plt.hist(data, bins)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Sample count")
    plt.tight_layout()
    viz.finalise(output)


def runtimes_range(db, output=None, where=None, nbins=25,
                   iqr=(0.25,0.75), figsize=None,
                   title="Normalised distribution of min and max runtimes"):
    data = [t[2:] for t in db.min_max_runtimes(where=where)]
    min_t, max_t = zip(*data)

    lower = labmath.filter_iqr(min_t, *iqr)
    upper = labmath.filter_iqr(max_t, *iqr)

    min_data = np.r_[lower, upper].min()
    max_data = np.r_[lower, upper].max()
    bins = np.linspace(min_data, max_data, nbins)

    plt.hist(lower, bins, label="Min")
    plt.hist(upper, bins, label="Max");
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Runtime (normalised to mean)")
    plt.legend(frameon=True)
    plt.tight_layout()
    if figsize is not None:
        plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)


def max_speedups(db, output=None):
    Speedups = sorted(db.max_speedups().values(), reverse=True)
    X = np.arange(len(Speedups))
    plt.plot(X, Speedups, 'b')
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=10)
    plt.axhline(y=1, color="k")
    plt.title("Max attainable speedups")
    plt.ylabel("Max speedup")
    plt.xlabel("Scenarios")
    plt.tight_layout()
    viz.finalise(output)


def num_samples(db, output=None, nbins=25):
    data = sorted([t[2] for t in db.num_samples()])
    bins = np.linspace(min(data), max(data), nbins)
    plt.hist(data, bins)
    plt.title("Sample counts for unique scenarios and params")
    plt.ylabel("Frequency")
    plt.xlabel("Number of samples")
    plt.tight_layout()
    viz.finalise(output)
