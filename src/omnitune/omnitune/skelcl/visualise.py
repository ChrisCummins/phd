#!/usr/bin/env python2

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

import labm8 as lab
from labm8 import io
from labm8 import fs
from labm8 import math as labmath

def _finalise(output=None):
    """
    Finalise a plot.

    Display or show the plot, then close it.

    Arguments:

        output (str): If out is not none, save figure at path
          output. Else, display the plot.
    """
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
        io.info("Wrote '{}'".format(output))
    plt.close()

def oracle_wgsizes_heatmap(db):
    fs.mkdir("img/oracle/devices")
    fs.mkdir("img/oracle/kernels")
    fs.mkdir("img/oracle/datasets")

    space = db.oracle_param_space()
    space.heatmap("img/oracle/heatmap.png",
                  title="All data")

    for i,device in enumerate(db.devices):
        io.info("Device heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE device='{0}')"
                 .format(device))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/devices/{0}.png"
                      .format(i), title=device)

    for kernel,ids in db.lookup_named_kernels().iteritems():
        io.info("Kernel heatmap", kernel, "...")
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN (SELECT id FROM scenarios WHERE "
                 "kernel IN ({0}))".format(",".join(id_wrapped)))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/kernels/{0}.png"
                      .format(kernel), title=kernel)

    for i,dataset in enumerate(db.datasets):
        io.info("Dataset heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE dataset='{0}')"
                 .format(dataset))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/datasets/{0}.png"
                      .format(i), title=dataset)


def max_wgsizes_heatmap(db, output=None):
    space = db.max_wgsize_space()
    space.heatmap(output=output, title="Distribution of maximum workgroup sizes")


def coverage(db, output=None, where=None, title="All data"):
    space = db.param_coverage_space(where=where)
    space.heatmap(output=output, title=title, vmin=0, vmax=1)


def safety(db, output=None, where=None, title="All data"):
    space = db.param_safe_space(where=where)
    space.heatmap(output=output, title=title, vmin=0, vmax=1)


def performance_vs_coverage(db, output=None):
    data = sorted([
        (
            db.perf_param_avg(param) * 100,
            db.perf_param_avg_legal(param) * 100,
            db.param_coverage(param) * 100
        )
        for param in db.params
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
    plt.title("Workgroup size performance vs. legality")
    plt.ylabel("Performance / Legality")
    plt.xlabel("Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    _finalise(output)


def performance_vs_max_wgsize(db, output=None):
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
    plt.title("Workgroup size performance vs. maximum workgroup size")
    plt.ylabel("Performance / Ratio max wgsize")
    plt.xlabel("Scenarios, Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    _finalise(output)


def kernel_performance(db, output=None):
    # Performance of all params across kernels.
    names = db.kernel_names
    Y = [db.performance_of_kernels_with_name(name) for name in names]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(names, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across kernels")
    _finalise(output)


def device_performance(db, output=None):
    devices = db.cpus + db.gpus
    Y = [db.performance_of_device(device) for device in devices]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(devices, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across devices")
    _finalise(output)

def dataset_performance(db, output=None):
    datasets = db.datasets
    Y = [db.performance_of_dataset(dataset) for dataset in datasets]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(datasets, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across datasets")
    _finalise(output)

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
    _finalise(output)


def min_max_runtimes(db, output=None, iqr=(0.25,0.75), nbins=25):
    data = [t[2:] for t in db.min_max_runtimes()]
    min_t, max_t = zip(*data)

    lower = labmath.filter_iqr(min_t, *iqr)
    upper = labmath.filter_iqr(max_t, *iqr)

    min_data = np.r_[lower, upper].min()
    max_data = np.r_[lower, upper].max()
    bins = np.linspace(min_data, max_data, nbins)

    plt.hist(lower, bins, label="Min")
    plt.hist(upper, bins, label="Max");
    plt.title("Normalised distribution of min and max runtimes")
    plt.ylabel("Frequency")
    plt.xlabel("Runtime (normalised to mean)")
    plt.legend(frameon=True)
    plt.tight_layout()
    _finalise(output)


def num_samples(db, output=None, nbins=25):
    data = sorted([t[2] for t in db.num_samples()])
    bins = np.linspace(min(data), max(data), nbins)
    plt.hist(data, bins)
    plt.title("Sample counts for unique scenarios and params")
    plt.ylabel("Frequency")
    plt.xlabel("Number of samples")
    plt.tight_layout()
    _finalise(output)
