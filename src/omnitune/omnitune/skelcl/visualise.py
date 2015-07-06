#!/usr/bin/env python2

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from . import space as _space

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8 import text
from labm8 import viz
from labm8 import prof


def num_samples(db, output=None, sample_range=None):
    def _get_sample_count_range():
        return db.execute(
            "SELECT\n"
            "    MIN(num_samples),\n"
            "    MAX(num_samples) + 2\n"
            "FROM runtime_stats"
        ).fetchone()

    # Range of sample counts.
    sample_range = sample_range or _get_sample_count_range()
    sequence = range(*sample_range)
    # Total number of test cases.
    num_instances = db.num_rows("runtime_stats")

    # Number of samples vs. ratio of runtime_stats.
    X,Y = zip(*[
        (i, db.execute(
            "SELECT\n"
            "    (CAST(Count(*) AS FLOAT) / CAST(? AS FLOAT)) * 100\n"
            "FROM runtime_stats\n"
            "WHERE num_samples >= ?",
            (num_instances, i)
        ).fetchone()[0])
        for i in sequence
    ])

    plt.title("Frequency of number of samples counts")
    plt.xlabel("Number of samples")
    plt.ylabel("Ratio of instances")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.plot(X, Y)
    plt.tight_layout()
    plt.xlim(*sample_range)
    viz.finalise(output)


def runtimes_variance(db, output=None, min_samples=1, where=None):
    # Create temporary table of scenarios and params to use, ignoring
    # those with less than "min_samples" samples.
    if "_temp" in db.tables:
        db.drop_table("_temp")

    db.execute("CREATE TABLE _temp (\n"
               "    scenario TEXT,\n"
               "    params TEXT,\n"
               "    PRIMARY KEY (scenario,params)\n"
               ")")
    query = (
        "INSERT INTO _temp\n"
        "SELECT\n"
        "    scenario,\n"
        "    params\n"
        "FROM runtime_stats\n"
        "WHERE num_samples >= ?"
    )
    if where is not None:
        query += " AND " + where
    db.execute(query, (min_samples,))

    X,Y = zip(*sorted([
        row for row in
        db.execute(
            "SELECT\n"
            "    AVG(runtime),\n"
            "    CONFERROR(runtime, .95) / AVG(runtime)\n"
            "FROM _temp\n"
            "LEFT JOIN runtimes\n"
            "    ON _temp.scenario=runtimes.scenario\n"
            "       AND _temp.params=runtimes.params\n"
            "GROUP BY _temp.scenario,_temp.params"
        )
    ], key=lambda x: x[0]))
    db.execute("DROP TABLE _temp")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    ax.set_xscale("log")

    plt.title("Runtime variance as a function of mean runtime")
    plt.ylabel("Normalised confidence interval")
    plt.xlabel("Runtime (ms)")
    plt.xlim(0, X[-1])
    plt.ylim(ymin=0)
    plt.tight_layout()
    viz.finalise(output)


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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Speedups, 'b')
    ax.set_yscale("log")
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.title("Max attainable speedups")
    plt.ylabel("Max speedup")
    plt.xlabel("Scenarios")
    plt.tight_layout()
    viz.finalise(output)


def classifier_speedups(db, classifier, output=None, sort=False,
                        job="xval_classifiers", **kwargs):
    """
    Plot speedup over the baseline of a classifier for each err_fn.
    """
    for err_fn in db.err_fns:
        performances = [row for row in
                        db.execute("SELECT speedup\n"
                                   "FROM classification_results\n"
                                   "WHERE job=? AND classifier=? AND err_fn=?",
                                   (job, classifier, err_fn))]
        if sort: performances = sorted(performances, reverse=True)
        plt.plot(performances, "-", label=err_fn)

    basename = ml.classifier_basename(classifier)
    plt.title(basename)
    plt.ylabel("Speedup")
    plt.xlabel("Test instances")
    plt.axhline(y=1, color="k")
    plt.xlim(xmin=0, xmax=len(performances))
    plt.legend()
    plt.tight_layout()
    viz.finalise(output)


def err_fn_speedups(db, err_fn, output=None, sort=False,
                    job="xval_classifiers", **kwargs):
    """
    Plot speedup over the baseline of all classifiers for an err_fn.
    """
    for classifier in db.classification_classifiers:
        basename = ml.classifier_basename(classifier)
        performances = [row for row in
                        db.execute("SELECT speedup\n"
                                   "FROM classification_results\n"
                                   "WHERE job=? AND classifier=? AND err_fn=?",
                                   (job, classifier, err_fn))]
        if sort: performances = sorted(performances, reverse=True)
        plt.plot(performances, "-", label=basename)

    plt.title(err_fn)
    plt.ylabel("Speedup")
    plt.xlabel("Test instances")
    plt.axhline(y=1, color="k")
    plt.xlim(xmin=0, xmax=len(performances))
    plt.legend()
    plt.tight_layout()
    viz.finalise(output)


def classification(db, output=None, job="xval_classifiers", **kwargs):
    err_fns = db.err_fns
    base_err_fn = err_fns[0]
    # Get a list of classifiers and result counts.
    query = db.execute(
        "SELECT classifier,Count(*) AS count\n"
        "FROM classification_results\n"
        "WHERE job=? AND err_fn=?\n"
        "GROUP BY classifier",
        (job,base_err_fn)
    )
    results = []
    for classifier,count in query:
        basename = ml.classifier_basename(classifier)
        correct, invalid = db.execute(
            "SELECT\n"
            "    (SUM(correct) / CAST(? AS FLOAT)) * 100,\n"
            "    (SUM(invalid) / CAST(? AS FLOAT)) * 100\n"
            "FROM classification_results\n"
            "WHERE job=? AND classifier=? AND err_fn=?",
            (count, count, job, classifier, base_err_fn)
        ).fetchone()
        # Get a list of mean speedups for each err_fn.
        speedups = [
            db.execute(
                "SELECT\n"
                "    GEOMEAN(speedup) * 100,\n"
                "    CONFERROR(speedup, .95) * 100\n"
                "FROM classification_results\n"
                "WHERE job=? AND classifier=? AND err_fn=?",
                (job, classifier, err_fn)
            ).fetchone()
            for err_fn in err_fns
        ]

        results.append([basename, correct, invalid] + speedups)

    # Zip into lists.
    labels, correct, invalid = zip(*[
        (text.truncate(result[0], 40), result[1], result[2])
        for result in results
    ])

    X = np.arange(len(labels))
    # Bar width.
    width = (.8 / (len(results[0]) - 1))

    plt.bar(X, invalid, width=width,
            color=sns.color_palette("Reds", 1), label="Invalid")
    plt.bar(X + width, correct, width=width,
            color=sns.color_palette("Blues", 1), label="Accuracy")
    # Colour palette for speedups.
    colors=sns.color_palette("Greens", len(err_fns))
    # Plot speedups.
    for i,err_fn in enumerate(db.err_fns):
        pairs = [result[3 + i] for result in results]
        speedups, yerrs = zip(*pairs)
        plt.bar(X + (2 + i) * width, speedups, width=width,
                label="Speedup ({})".format(err_fn), color=colors[i])

        # Plot confidence intervals separately so that we can have
        # full control over formatting.
        _,caps,_ = plt.errorbar(X + (2.5 + i) * width, speedups, fmt="none",
                                yerr=yerrs, capsize=3, ecolor="k")
        for cap in caps:
            cap.set_color('k')
            cap.set_markeredgewidth(1)

    plt.xlim(xmin=-.2)
    plt.xticks(X + .4, labels)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.title("Classification results")

    # Add legend *beneath* plot. To do this, we need to pass some
    # extra arguments to plt.savefig(). See:
    #
    # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
    #
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
    viz.finalise(output, additional_artists=art, bbox_inches="tight")


def xval_runtime_regression(db, output=None, job="xval_runtimes", **kwargs):
    """
    Plot accuracy of a classifier at predicted runtime.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for classifier in db.regression_classifiers:
        basename = ml.classifier_basename(classifier)
        actual, norm_predicted = zip(*sorted([
            row for row in
            db.execute(
                "SELECT\n"
                "    actual,\n"
                "    predicted\n"
                "FROM runtime_regression_results\n"
                "WHERE job=? AND classifier=?",
                (job, classifier)
            )
        ], key=lambda x: x[0], reverse=True))

        ax.plot(norm_predicted, label=basename)
        # TODO: Once we have all regression results, we only need to
        # plot "actual" once.
        ax.plot(actual, label=basename + " - Actual")
    ax.set_yscale("log")
    plt.legend()
    plt.title("Runtime")
    plt.xlabel("Test instances (sorted by descending runtime)")
    plt.ylabel("Runtime (ms)")
    viz.finalise(output)
