#!/usr/bin/env python2

from __future__ import division

import operator
import re

from collections import defaultdict
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from . import space as _space
from . import unhash_params

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8 import text
from labm8 import viz
from labm8 import prof


def num_samples(db, output=None, sample_range=None, **kwargs):

    # Range of sample counts.
    sample_range = sample_range or (1, 100)

    num_instances = db.num_rows("runtime_stats")

    X = np.arange(num_instances)
    Y = np.zeros(num_instances)

    for i in range(sample_range[0], sample_range[1] + 1):
        Y[i] = db.execute("SELECT (Count(*) * 1.0 / ?) * 100 "
                          "FROM runtime_stats WHERE num_samples >= ?",
                          (num_instances, i)).fetchone()[0]

    title = kwargs.pop("title", "Frequency of number of samples counts")
    plt.title(title)
    plt.xlabel("Number of samples")
    plt.ylabel("Ratio of instances")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.plot(X, Y)
    plt.xlim(*sample_range)
    viz.finalise(output, **kwargs)


def runtimes_variance(db, output=None, min_samples=1, where=None, **kwargs):
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

    title = kwargs.pop("title",
                       "Runtime variance as a function of mean runtime")
    plt.title(title)
    plt.ylabel("Normalised confidence interval")
    plt.xlabel("Runtime (ms)")
    plt.xlim(0, X[-1])
    plt.ylim(ymin=0)
    viz.finalise(output, **kwargs)


def max_wgsizes(db, output=None, trisurf=False, **kwargs):
    space = db.max_wgsize_space()
    space.clip(50, 50)
    if "title" not in kwargs:
        kwargs["title"] = "Distribution of maximum workgroup sizes"
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
    space.clip(50, 50)
    if "title" not in kwargs: kwargs["title"] = "All data"
    if trisurf:
        space.trisurf(output=output, **kwargs)
    else:
        if "vim" not in kwargs: kwargs["vmin"] = 0
        if "vmax" not in kwargs: kwargs["vmax"] = 1
        space.heatmap(output=output, **kwargs)


def scenario_performance(db, scenario, output=None, title=None, type="heatmap",
                         reshape_args=None):
    space = _space.ParamSpace.from_dict(db.perf_scenario(scenario))
    if reshape_args is not None:
        space.reshape(**reshape_args)
    if type == "heatmap":
        space.heatmap(output=output, title=title)
    elif type == "trisurf":
        space.trisurf(output=output, title=title, zlabel="Performance",
                      rotation=45)
    elif type == "bar3d":
        space.bar3d(output=output, title=title, zlabel="Performance",
                    rotation=45)
    else:
        raise viz.Error("Unrecognised visualisation type '{}'".format(type))



def performance_vs_coverage(db, output=None, max_values=200, **kwargs):
    data = [
        row for row in
        db.execute(
            "SELECT "
            "    CASE WHEN coverage=1 THEN performance * 100.0 ELSE 0 END AS perf_all, "
            "    performance * 100.0 AS perf_legal, "
            "    coverage * 100.0 "
            "FROM param_stats "
            "ORDER BY coverage DESC, perf_all DESC, perf_legal DESC "
            "LIMIT ?",
            (max_values,)
        )
    ]
    X = np.arange(len(data))

    GeoPerformance, Performance, Coverage = zip(*data)

    ax = plt.subplot(111)
    ax.plot(X, Coverage, 'r', label="Legality")
    ax.plot(X, Performance, 'g', label="Performance (when legal)")
    ax.plot(X, GeoPerformance, 'b', label="Performance")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    title = kwargs.pop("title", "Workgroup size performance vs. legality")
    plt.title(title)
    plt.xlabel("Parameters (sorted by legality and performance)")
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, 1.2), ncol=3)]
    viz.finalise(output, additional_artists=art, **kwargs)


def oracle_speedups(db, output=None, **kwargs):
    data = db.oracle_speedups().values()
    #Speedups = sorted(data, reverse=True)
    Speedups = data
    X = np.arange(len(Speedups))

    plt.plot(X, Speedups)
    plt.xlim(0, len(X) - 1)
    title = kwargs.pop("title", "Attainable performance over baseline")
    plt.title(title)
    plt.xlabel("Scenarios")
    plt.ylabel("Speedup")
    viz.finalise(output, **kwargs)


def num_params_vs_accuracy(db, output=None, where=None, **kwargs):
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
    title = kwargs.pop("title", "Number of workgroup sizes vs. oracle accuracy")
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of distinct workgroup sizes")
    plt.legend(frameon=True)
    viz.finalise(output, **kwargs)


def _performance_size(data, output, title, xlabel, percs=True, **kwargs):
    # Calculate moving average.
    avgs = defaultdict(list)
    for p,s in data:
        avgs[s].append(p)
    for s,p in avgs.items():
        mean = labmath.mean(p)
        avgs[s] = (mean, labmath.confinterval(p, array_mean=mean,
                                              error_only=True) * 2)
    avgs = sorted(avgs.items())

    Y, X = zip(*data)
    aX, data = zip(*avgs)
    aY, aErr = zip(*data)

    plt.scatter(X, Y)
    plt.errorbar(aX, aY, yerr=aErr, color="r")
    if percs:
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d%%'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(0, max(X) + 2)
    plt.ylim(0, max(Y) + 2)
    plt.title(title)
    plt.ylabel("Performance (% oracle)")
    plt.xlabel(xlabel)
    viz.finalise(output, **kwargs)


def performance_vs_max_wgsize(db, output=None, **kwargs):
    # TODO: Replace with box plot of 10% ranges of max wgsize.
    data = [
        row for row in
        db.execute(
            "SELECT "
            "    (scenario_stats.oracle_runtime / runtime_stats.mean) * 100 "
            "  AS performance, "
            "    (params.wg_c * params.wg_r * 1.0 / kernels.max_wg_size) "
            "  * 100 AS ratio_max_wgsize "
            "FROM runtime_stats "
            "LEFT JOIN scenarios "
            "  ON runtime_stats.scenario=scenarios.id "
            "LEFT JOIN kernels "
            "  ON scenarios.kernel=kernels.id "
            "LEFT JOIN params "
            "  ON runtime_stats.params=params.id "
            "LEFT JOIN scenario_stats "
            "  ON runtime_stats.scenario=scenario_stats.scenario "
            "ORDER BY RANDOM() "
            "LIMIT 1000"
        )
    ]

    title = kwargs.pop("title",
                       "Workgroup size performance vs. maximum workgroup size")
    _performance_size(data, output, title, "Workgroup size (% max)", **kwargs)


def performance_vs_wgsize(db, output=None, **kwargs):
    data = [
        (
            db.perf(scenario, param) * 100,
            reduce(operator.mul, unhash_params(param), 1)
        )
        for scenario, param in db.scenario_params
    ]

    title = kwargs.pop("title",
                       "Size of workgroup vs. performance")
    _performance_size(data, output, title, "Workgroup size", percs=False,
                      **kwargs)


def performance_vs_wg_c(db, output=None, **kwargs):
    data = [
        row for row in
        db.execute(
            "SELECT "
            "    scenario_stats.oracle_runtime / runtime_stats.mean "
            "  AS performance, "
            "    params.wg_c "
            "FROM runtime_stats "
            "LEFT JOIN scenarios "
            "  ON runtime_stats.scenario=scenarios.id "
            "LEFT JOIN kernels "
            "  ON scenarios.kernel=kernels.id "
            "LEFT JOIN params "
            "  ON runtime_stats.params=params.id "
            "LEFT JOIN scenario_stats "
            "  ON runtime_stats.scenario=scenario_stats.scenario "
            "ORDER BY RANDOM() "
            "LIMIT 1000"
        )
    ]

    title = kwargs.pop("title",
                       "Workgroup size performance vs. number of columns")
    _performance_size(data, output, title, "Workgroup columns", percs=False,
                      **kwargs)


def performance_vs_wg_r(db, output=None, **kwargs):
    data = [
        row for row in
        db.execute(
            "SELECT "
            "    scenario_stats.oracle_runtime / runtime_stats.mean "
            "  AS performance, "
            "    params.wg_r "
            "FROM runtime_stats "
            "LEFT JOIN scenarios "
            "  ON runtime_stats.scenario=scenarios.id "
            "LEFT JOIN kernels "
            "  ON scenarios.kernel=kernels.id "
            "LEFT JOIN params "
            "  ON runtime_stats.params=params.id "
            "LEFT JOIN scenario_stats "
            "  ON runtime_stats.scenario=scenario_stats.scenario "
            "ORDER BY RANDOM() "
            "LIMIT 1000"
        )
    ]

    title = kwargs.pop("title",
                       "Workgroup size performance vs. number of rows")
    _performance_size(data, output, title, "Workgroup rows", percs=False,
                      **kwargs)


def _performance_plot(output, labels, values, title, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(data=values)
    ax.set_xticklabels(labels, rotation=90)
    plt.ylim(ymin=0, ymax=1)
    plt.title(title)
    viz.finalise(output, **kwargs)


def kernel_performance(db, output=None, **kwargs):
    labels = ["synthetic"] + db.real_kernel_names

    values = [lab.flatten([db.performance_of_kernels_with_name(name) for name in
                          db.synthetic_kernel_names])]
    values += [db.performance_of_kernels_with_name(name) for name in
               db.real_kernel_names]

    title = kwargs.pop("title", "Workgroup size performance across kernels")
    _performance_plot(output, labels, values, title, **kwargs)


def device_performance(db, output=None, **kwargs):
    def _fmtid(id):
        name = id.strip()
        name = re.sub("^\dx ", "", name)
        name = re.sub("GeForce", "Nvidia", name)
        name = re.sub("Tahiti", "AMD Tahiti 7970", name)
        name = re.sub("Intel\(R\) Core\(TM\)", "Intel", name)
        name = re.sub(" CPU @ [0-9\.]+GHz", "", name)
        return name

    ids = db.cpus + db.gpus # Arrange CPUs on the left, GPUs on the right.
    labels = [_fmtid(id) for id in ids]
    values = [db.performance_of_device(id) for id in ids]
    title = kwargs.pop("title", "Workgroup size performance across devices")
    _performance_plot(output, labels, values, title, **kwargs)


def dataset_performance(db, output=None, **kwargs):
    labels = db.datasets
    values = [db.performance_of_dataset(label) for label in labels]
    title = kwargs.pop("title", "Workgroup size performance across datasets")
    _performance_plot(output, labels, values, title, **kwargs)


def runtimes_range(db, output=None, where=None, nbins=25,
                   iqr=(0.25,0.75), **kwargs):
    # data = [t[2:] for t in db.min_max_runtimes(where=where)]
    # min_t, max_t = zip(*data)

    # lower = labmath.filter_iqr(min_t, *iqr)
    # upper = labmath.filter_iqr(max_t, *iqr)

    # min_data = np.r_[lower, upper].min()
    # max_data = np.r_[lower, upper].max()
    # bins = np.linspace(min_data, max_data, nbins)

    # Plt.hist(lower, bins, label="Min")
    # plt.hist(upper, bins, label="Max");
    title = kwargs.pop("title", "Normalised distribution of min and max runtimes")
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Runtime (normalised to mean)")
    plt.legend(frameon=True)
    viz.finalise(output, **kwargs)


def max_speedups(db, output=None, **kwargs):
    Speedups = db.max_speedups().values()
    X = np.arange(len(Speedups))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Speedups)
    plt.xlim(xmin=0, xmax=len(X) - 1)
    title = kwargs.pop("title", "Max attainable speedups")
    plt.title(title)
    plt.ylabel("Max speedup")
    plt.xlabel("Scenarios")
    viz.finalise(output, **kwargs)


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
    viz.finalise(output, **kwargs)


def err_fn_speedups(db, err_fn, output=None, sort=False,
                    job="xval", **kwargs):
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

    title = kwargs.pop("title", err_fn)
    plt.title(title)
    plt.ylabel("Speedup")
    plt.xlabel("Test instances")
    plt.axhline(y=1, color="k")
    plt.xlim(xmin=0, xmax=len(performances))
    plt.legend()
    viz.finalise(output, **kwargs)


def err_fn_performance(db, output=None, job="xval", **kwargs):
    err_fns = db.err_fns
    results = [
        db.execute(
            "SELECT\n"
            "    GEOMEAN(performance) * 100,\n"
            "    CONFERROR(performance, .95) * 100,\n"
            "    GEOMEAN(speedup) * 100,\n"
            "    CONFERROR(speedup, .95) * 100\n"
            "FROM classification_results\n"
            "WHERE job=? AND err_fn=? AND invalid=1",
            (job, err_fn)
        ).fetchone()
        for err_fn in err_fns
    ]

    perfs, perfErrors, speedups, speedupErrors = zip(*results)

    X = np.arange(len(err_fns))
    # Bar width.
    width = (.8 / (len(results[0]) - 1))

    plt.bar(X, perfs, width=width,
            color=sns.color_palette("Reds", 1), label="Performance")
    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _,caps,_ = plt.errorbar(X + .5 * width, perfs, fmt="none",
                            yerr=perfErrors, capsize=3, ecolor="k")
    for cap in caps:
        cap.set_color('k')
        cap.set_markeredgewidth(1)

    plt.bar(X + width, speedups, width=width,
            color=sns.color_palette("Greens", 1), label="Speedup")
    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _,caps,_ = plt.errorbar(X + 1.5 * width, speedups, fmt="none",
                            yerr=speedupErrors, capsize=3, ecolor="k")
    for cap in caps:
        cap.set_color('k')
        cap.set_markeredgewidth(1)

    plt.xlim(xmin=-.2)
    plt.xticks(X + .4, err_fns)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))

    title = kwargs.pop("title", "Error handler performance for " + job)
    plt.title(title)

    # Add legend *beneath* plot. To do this, we need to pass some
    # extra arguments to plt.savefig(). See:
    #
    # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
    #
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
    viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)


def classification(db, output=None, job="xval", **kwargs):
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

    title = kwargs.pop("title", "Classification results for " + job)
    plt.title(title)

    # Add legend *beneath* plot. To do this, we need to pass some
    # extra arguments to plt.savefig(). See:
    #
    # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
    #
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
    viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)


def runtime_regression(db, output=None, job="xval", **kwargs):
    """
    Plot accuracy of a classifier at predicted runtime.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = sns.color_palette()
    i, actual = 0, []

    for i,classifier in enumerate(db.regression_classifiers):
        basename = ml.classifier_basename(classifier)
        actual, predicted = zip(*sorted([
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

        if basename == "ZeroR":
            ax.plot(predicted, label=basename, color=colors[i - 1])
        else:
            ax.scatter(np.arange(len(predicted)), predicted, label=basename,
                       color=colors[i - 1])

    ax.plot(actual, label="Actual", color=colors[i])
    ax.set_yscale("log")
    plt.xlim(0, len(actual))
    plt.legend()
    title = kwargs.pop("title", "Runtime regression for " + job)
    plt.title(title)
    plt.xlabel("Test instances (sorted by descending runtime)")
    plt.ylabel("Runtime (ms, log)")
    viz.finalise(output, **kwargs)


def runtime_classification(db, output=None, job="xval", **kwargs):
    """
    Plot performance of classification using runtime regression.
    """
    # Get a list of classifiers and result counts.
    query = db.execute(
        "SELECT classifier,Count(*) AS count\n"
        "FROM runtime_classification_results\n"
        "WHERE job=? GROUP BY classifier", (job,)
    )
    results = []
    for classifier,count in query:
        basename = ml.classifier_basename(classifier)
        correct = db.execute(
            "SELECT\n"
            "    (SUM(correct) / CAST(? AS FLOAT)) * 100\n"
            "FROM runtime_classification_results\n"
            "WHERE job=? AND classifier=?",
            (count, job, classifier)
        ).fetchone()[0]
        # Get a list of mean speedups for each err_fn.
        speedups = [
            row for row in
            db.execute(
                "SELECT\n"
                "    GEOMEAN(speedup) * 100,\n"
                "    CONFERROR(speedup, .95) * 100\n"
                "FROM runtime_classification_results\n"
                "WHERE job=? AND classifier=?",
                (job, classifier)
            ).fetchone()
        ]

        results.append([basename, correct] + speedups)

    # Zip into lists.
    labels, correct, speedups, yerrs = zip(*results)

    X = np.arange(len(labels))
    # Bar width.
    width = (.8 / (len(results[0]) - 1))

    plt.bar(X + width, correct, width=width,
            color=sns.color_palette("Blues", 1), label="Accuracy")
    plt.bar(X + 2 * width, speedups, width=width,
            color=sns.color_palette("Greens", 1), label="Speedup")
    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _,caps,_ = plt.errorbar(X + 2.5 * width, speedups, fmt="none",
                            yerr=yerrs, capsize=3, ecolor="k")
    for cap in caps:
        cap.set_color('k')
        cap.set_markeredgewidth(1)

    plt.xlim(xmin=-.2)
    plt.xticks(X + .4, labels)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))

    title = kwargs.pop("title",
                       "Classification results for " + job +
                       " using runtime regression")
    plt.title(title)

    # Add legend *beneath* plot. To do this, we need to pass some
    # extra arguments to plt.savefig(). See:
    #
    # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
    #
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
    viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)


def speedup_regression(db, output=None, job="xval", **kwargs):
    """
    Plot accuracy of a classifier at predicted runtime.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = sns.color_palette()
    for i,classifier in enumerate(db.regression_classifiers):
        basename = ml.classifier_basename(classifier)
        actual, predicted = zip(*sorted([
            row for row in
            db.execute(
                "SELECT\n"
                "    actual,\n"
                "    predicted\n"
                "FROM speedup_regression_results\n"
                "WHERE job=? AND classifier=?",
                (job, classifier)
            )
        ], key=lambda x: x[0], reverse=True))

        if basename == "ZeroR":
            ax.plot(predicted, label=basename, color=colors[i - 1])
        else:
            ax.scatter(np.arange(len(predicted)), predicted, label=basename,
                       color=colors[i - 1])

    ax.plot(actual, label="Actual", color=colors[i])
    # ax.set_yscale("log")
    plt.xlim(0, len(actual))
    plt.legend()
    title = kwargs.pop("title", "Speedup regression for " + job)
    plt.title(title)
    plt.xlabel("Test instances (sorted by descending speedup)")
    plt.ylabel("Speedup")
    viz.finalise(output, **kwargs)


def speedup_classification(db, output=None, job="xval", **kwargs):
    """
    Plot performance of classification using speedup regression.
    """
    # Get a list of classifiers and result counts.
    query = db.execute(
        "SELECT classifier,Count(*) AS count\n"
        "FROM speedup_classification_results\n"
        "WHERE job=? GROUP BY classifier", (job,)
    )
    results = []
    for classifier,count in query:
        basename = ml.classifier_basename(classifier)
        correct = db.execute(
            "SELECT\n"
            "    (SUM(correct) / CAST(? AS FLOAT)) * 100\n"
            "FROM speedup_classification_results\n"
            "WHERE job=? AND classifier=?",
            (count, job, classifier)
        ).fetchone()[0]
        # Get a list of mean speedups for each err_fn.
        speedups = [
            row for row in
            db.execute(
                "SELECT\n"
                "    GEOMEAN(speedup) * 100,\n"
                "    CONFERROR(speedup, .95) * 100\n"
                "FROM speedup_classification_results\n"
                "WHERE job=? AND classifier=?",
                (job, classifier)
            ).fetchone()
        ]

        results.append([basename, correct] + speedups)

    # Zip into lists.
    labels, correct, speedups, yerrs = zip(*results)

    X = np.arange(len(labels))
    # Bar width.
    width = (.8 / (len(results[0]) - 1))

    plt.bar(X + width, correct, width=width,
            color=sns.color_palette("Blues", 1), label="Accuracy")
    plt.bar(X + 2 * width, speedups, width=width,
            color=sns.color_palette("Greens", 1), label="Speedup")
    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _,caps,_ = plt.errorbar(X + 2.5 * width, speedups, fmt="none",
                            yerr=yerrs, capsize=3, ecolor="k")
    for cap in caps:
        cap.set_color('k')
        cap.set_markeredgewidth(1)

    plt.xlim(xmin=-.2)
    plt.xticks(X + .4, labels)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))

    title = kwargs.pop("title",
                       "Classification results for " + job +
                       " using speedup regression")
    plt.title(title)

    # Add legend *beneath* plot. To do this, we need to pass some
    # extra arguments to plt.savefig(). See:
    #
    # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
    #
    art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
    viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)
