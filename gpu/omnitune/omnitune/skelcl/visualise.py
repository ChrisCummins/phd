#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from phd import labm8 as lab

from . import space as _space
from labm8.py import math as labmath
from labm8.py import ml
from labm8.py import text
from labm8.py import viz


def fmtdevid(id):
  name = id.strip()
  name = re.sub("^\dx", "", name)
  name = re.sub("GeForce", "Nvidia", name)
  name = re.sub("Tahiti", "AMD Tahiti 7970", name)
  name = re.sub("Intel\(R\) Core\(TM\)", "Intel", name)
  name = re.sub(" CPU @ [0-9\.]+GHz", "", name)
  return name


def num_samples(db, output=None, sample_range=None, **kwargs):
  # Range of sample counts.
  sample_range = sample_range or (1, 100)

  num_instances = db.num_rows("runtime_stats")

  X = np.arange(num_instances)
  Y = np.zeros(num_instances)

  for i in range(sample_range[0], sample_range[1] + 1):
    Y[i] = db.execute(
      "SELECT (Count(*) * 1.0 / ?) * 100 "
      "FROM runtime_stats WHERE num_samples >= ?",
      (num_instances, i),
    ).fetchone()[0]

  title = kwargs.pop("title", "Frequency of number of samples counts")
  plt.title(title)
  plt.xlabel("Sample count")
  plt.ylabel("Ratio of test cases")
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  plt.plot(X, Y)
  plt.xlim(*sample_range)
  viz.finalise(output, **kwargs)


def num_params(db, output=None, sample_range=None, **kwargs):
  # Range of param counts.
  sample_range = sample_range or (1, 100)

  num_instances = db.num_rows("scenario_stats")

  X = np.arange(num_instances)
  Y = np.zeros(num_instances)

  for i in range(sample_range[0], sample_range[1] + 1):
    Y[i] = db.execute(
      "SELECT (Count(*) * 1.0 / ?) * 100 "
      "FROM scenario_stats WHERE num_params >= ?",
      (num_instances, i),
    ).fetchone()[0]

  title = kwargs.pop("title", "Parameter values count")
  plt.title(title)
  plt.xlabel("Number of parameters")
  plt.ylabel("Ratio of scenarios")
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  plt.plot(X, Y)
  plt.xlim(*sample_range)
  viz.finalise(output, **kwargs)


def runtimes_histogram(runtimes, output=None, color=None, **kwargs):
  mean = np.mean(runtimes)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  sns.distplot(runtimes, bins=40, kde_kws={"bw": 0.3}, color=color)

  ax.axvline(mean, color="0.25", linestyle="--")
  plt.xlim(min(runtimes), max(runtimes))
  plt.gca().axes.get_yaxis().set_ticks([])
  plt.xlabel("Runtime (ms)")
  plt.locator_params(axis="x", nbins=6)
  viz.finalise(output, **kwargs)


def confinterval_trend(
  sample_counts, confintervals, output=None, vlines=[], **kwargs
):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(sample_counts, [y * 100 for y in confintervals])
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  for vline in vlines:
    ax.axvline(vline, color="k", linestyle="--")
  plt.ylabel("95\\% CI / mean")
  plt.xlabel("Number of samples")
  plt.xlim(min(sample_counts), max(sample_counts))
  viz.finalise(output, **kwargs)


def runtimes_variance(db, output=None, min_samples=1, where=None, **kwargs):
  # Create temporary table of scenarios and params to use, ignoring
  # those with less than "min_samples" samples.
  if "_temp" in db.tables:
    db.drop_table("_temp")

  db.execute(
    "CREATE TABLE _temp (\n"
    "    scenario TEXT,\n"
    "    params TEXT,\n"
    "    PRIMARY KEY (scenario,params)\n"
    ")"
  )
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

  X, Y = zip(
    *sorted(
      [
        row
        for row in db.execute(
          "SELECT\n"
          "    AVG(runtime),\n"
          "    CONFERROR(runtime, .95) / AVG(runtime)\n"
          "FROM _temp\n"
          "LEFT JOIN runtimes\n"
          "    ON _temp.scenario=runtimes.scenario\n"
          "       AND _temp.params=runtimes.params\n"
          "GROUP BY _temp.scenario,_temp.params"
        )
      ],
      key=lambda x: x[0],
    )
  )
  db.execute("DROP TABLE _temp")

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X, Y)
  ax.set_xscale("log")

  title = kwargs.pop("title", "Runtime variance as a function of mean runtime")
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
    space.trisurf(output=output, zlabel="Coverage (ratio)", **kwargs)
  else:
    space.heatmap(output=output, **kwargs)


def coverage(db, output=None, trisurf=False, clip=100, **kwargs):
  if trisurf:
    data = {
      row[0]: row[1]
      for row in db.execute(
        "SELECT params,num_scenarios "
        "FROM param_stats "
        "LEFT JOIN params "
        "ON param_stats.params=params.id "
        "WHERE wg_c <= ? AND wg_r <= ?",
        (clip, clip),
      )
    }
    space = _space.ParamSpace.from_dict(data)
    space.log()
    space.trisurf(output=output, zlabel="Legal frequency (log)", **kwargs)
  else:
    data = {
      row[0]: row[1]
      for row in db.execute(
        "SELECT params,coverage "
        "FROM param_stats "
        "LEFT JOIN params "
        "ON param_stats.params=params.id "
        "WHERE wg_c <= ? AND wg_r <= ?",
        (clip, clip),
      )
    }
    space = _space.ParamSpace.from_dict(data)
    space.heatmap(output=output, **kwargs)


def performance(db, output=None, trisurf=False, clip=100, **kwargs):
  data = {
    row[0]: row[1]
    for row in db.execute(
      "SELECT params,performance "
      "FROM param_stats "
      "LEFT JOIN params "
      "ON param_stats.params=params.id "
      "WHERE wg_c <= ? AND wg_r <= ?",
      (clip, clip),
    )
  }

  space = _space.ParamSpace.from_dict(data)
  if trisurf:
    space.trisurf(output=output, rotation=45, **kwargs)
  else:
    space.heatmap(output=output, **kwargs)


def safety(db, output=None, where=None, **kwargs):
  space = db.param_safe_space(where=where)
  if "title" not in kwargs:
    kwargs["title"] = "All data"
  if "vim" not in kwargs:
    kwargs["vmin"] = 0
  if "vmax" not in kwargs:
    kwargs["vmax"] = 1
  space.heatmap(output=output, **kwargs)


def oracle_wgsizes(db, output=None, trisurf=False, clip=100, **kwargs):
  space = db.oracle_param_space(normalise=False)
  space.clip(clip, clip)
  space.log()
  if trisurf:
    space.trisurf(
      output=output, zlabel="Oracle frequency (log)", vmax=1.5, **kwargs
    )
  else:
    space.heatmap(output=output, **kwargs)


def scenario_performance(
  db, scenario, output=None, title=None, type="heatmap", reshape_args=None
):
  space = _space.ParamSpace.from_dict(db.perf_scenario(scenario))
  if reshape_args is not None:
    space.reshape(**reshape_args)
  if type == "heatmap":
    space.heatmap(output=output, title=title)
  elif type == "trisurf":
    space.trisurf(output=output, title=title, zlabel="Performance", rotation=45)
  elif type == "bar3d":
    space.bar3d(output=output, title=title, zlabel="Performance", rotation=45)
  else:
    raise viz.Error("Unrecognised visualisation type '{}'".format(type))


def performance_vs_coverage(db, output=None, max_values=250, **kwargs):
  data = [
    row
    for row in db.execute(
      "SELECT "
      "    performance AS performance, "
      "    coverage "
      "FROM param_stats"
    )
  ]
  frame = pandas.DataFrame(data, columns=("Performance", "Legality"))
  sns.jointplot("Legality", "Performance", data=frame, xlim=(0, 1), ylim=(0, 1))
  viz.finalise(output, **kwargs)


def oracle_speedups(db, output=None, **kwargs):
  data = db.oracle_speedups().values()
  # Speedups = sorted(data, reverse=True)
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
  freqs = sorted(
    db.oracle_param_frequencies(normalise=True).values(), reverse=True
  )
  acc = 0
  Data = [0] * len(freqs)
  for i, freq in enumerate(freqs):
    acc += freq * 100
    Data[i] = acc

  X = np.arange(len(Data))
  ax = plt.subplot(111)
  ax.plot(X, Data)
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  plt.xlim(xmin=0, xmax=len(X) - 1)
  plt.ylim(ymin=0, ymax=100)
  title = kwargs.pop("title", "Number of workgroup sizes vs. oracle accuracy")
  plt.title(title)
  plt.ylabel("Accuracy")
  plt.xlabel("Number of distinct workgroup sizes")
  plt.legend(frameon=True)
  viz.finalise(output, **kwargs)


def pie(data, output=None, **kwargs):
  labels, values = zip(*data)
  plt.pie(values, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
  viz.finalise(output, **kwargs)


def performance_vs_max_wgsize(ratios, output=None, color=None, **kwargs):
  title = kwargs.pop(
    "title", "Workgroup size performance vs. maximum workgroup size"
  )
  fig = plt.figure()
  ax = fig.add_subplot(111)

  sns.boxplot(data=ratios, linewidth=1, fliersize=1)
  # sns.violinplot(data=ratios, inner="quartile", linewidth=.5)

  multiplier = kwargs.pop("multiplier", 10)
  ax.set_xticklabels(
    [str((x + 1) * multiplier) + r"\%" for x in np.arange(len(ratios))]
  )

  title = kwargs.pop("title", "")
  plt.title(title)
  plt.ylim(ymin=0, ymax=1)
  plt.ylabel("Performance")
  xlabel = kwargs.pop("xlabel", "")
  plt.xlabel(xlabel)
  viz.finalise(output, **kwargs)


def _performance_plot(output, labels, values, title, color=None, **kwargs):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  sns.boxplot(data=values, linewidth=1, fliersize=1)
  # sns.violinplot(data=values, inner="quartile", linewidth=.5)

  ax.set_xticklabels(labels, rotation=90)
  plt.ylim(ymin=0, ymax=1)
  plt.ylabel("Performance")
  plt.title(title)
  viz.finalise(output, **kwargs)


def kernel_performance(db, output=None, **kwargs):
  labels = ["synthetic"] + db.real_kernel_names

  values = [
    lab.flatten(
      [
        db.performance_of_kernels_with_name(name)
        for name in db.synthetic_kernel_names
      ]
    )
  ]
  values += [
    db.performance_of_kernels_with_name(name) for name in db.real_kernel_names
  ]

  title = kwargs.pop("title", "Workgroup size performance across kernels")
  _performance_plot(
    output, labels, values, title, color=sns.color_palette("Greens"), **kwargs
  )


def device_performance(db, output=None, **kwargs):
  ids = db.cpus + db.gpus  # Arrange CPUs on the left, GPUs on the right.
  labels = [fmtdevid(id) for id in ids]
  values = [db.performance_of_device(id) for id in ids]
  title = kwargs.pop("title", "Workgroup size performance across devices")
  _performance_plot(
    output, labels, values, title, color=sns.color_palette("Blues"), **kwargs
  )


def dataset_performance(db, output=None, **kwargs):
  labels = db.datasets
  values = [db.performance_of_dataset(label) for label in labels]
  title = kwargs.pop("title", "Workgroup size performance across datasets")
  _performance_plot(
    output, labels, values, title, color=sns.color_palette("Reds"), **kwargs
  )


def runtimes_range(
  db, output=None, where=None, nbins=25, iqr=(0.25, 0.75), **kwargs
):
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
  max_speedups, min_static, he = zip(*db.max_and_static_speedups)
  X = np.arange(len(max_speedups))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(X, max_speedups, "r", linestyle="--", label="Max")
  ax.plot(X, min_static, label="$w_{(4 \\times 4)}$")
  ax.plot(X, he, linestyle="-", label="$w_{(32 \\times 4)}$")
  # plt.ylim(ymin=0, ymax=100)
  plt.xlim(xmin=0, xmax=len(X) - 1)
  title = kwargs.pop("title", "Max attainable speedups")
  plt.title(title)
  ax.set_yscale("log")
  plt.legend(frameon=True)
  plt.ylabel("Speedup (log)")
  plt.xlabel("Scenarios (sorted by descending max speedup)")
  viz.finalise(output, **kwargs)


def classifier_speedups(
  db, classifier, output=None, sort=False, job="xval_classifiers", **kwargs
):
  """
  Plot speedup over the baseline of a classifier for each err_fn.
  """
  for err_fn in db.err_fns:
    performances = [
      row
      for row in db.execute(
        "SELECT speedup\n"
        "FROM classification_results\n"
        "WHERE job=? AND classifier=? AND err_fn=?",
        (job, classifier, err_fn),
      )
    ]
    if sort:
      performances = sorted(performances, reverse=True)
    plt.plot(performances, "-", label=err_fn)

  basename = ml.classifier_basename(classifier)
  plt.title(basename)
  plt.ylabel("Speedup")
  plt.xlabel("Test instances")
  plt.axhline(y=1, color="k")
  plt.xlim(xmin=0, xmax=len(performances))
  plt.legend()
  viz.finalise(output, **kwargs)


def err_fn_speedups(db, err_fn, output=None, sort=False, job="xval", **kwargs):
  """
  Plot speedup over the baseline of all classifiers for an err_fn.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  for classifier in db.classification_classifiers:
    basename = ml.classifier_basename(classifier)
    performances = [
      row
      for row in db.execute(
        "SELECT speedup\n"
        "FROM classification_results\n"
        "WHERE job=? AND classifier=? AND err_fn=?",
        (job, classifier, err_fn),
      )
    ]
    if sort:
      performances = sorted(performances, reverse=True)
    plt.plot(performances, "-", label=basename)
  plt.plot([1 for _ in performances], "-", label="ZeroR")

  title = kwargs.pop("title", err_fn)
  ax.set_yscale("log")
  plt.title(title)
  plt.ylabel("Speedup (log)")
  plt.xlabel("Test instances")
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
      "WHERE job=? AND err_fn=? AND (illegal=1 or refused=1)",
      (job, err_fn),
    ).fetchone()
    for err_fn in err_fns
  ]

  perfs, perfErrors, speedups, speedupErrors = zip(*results)

  X = np.arange(len(err_fns))
  # Bar width.
  width = 0.8 / (len(results[0]) - 1)

  plt.bar(
    X,
    perfs,
    width=width,
    color=sns.color_palette("Reds", 1),
    label="Performance",
  )
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = plt.errorbar(
    X + 0.5 * width, perfs, fmt="none", yerr=perfErrors, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  plt.bar(
    X + width,
    speedups,
    width=width,
    color=sns.color_palette("Greens", 1),
    label="Speedup",
  )
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = plt.errorbar(
    X + 1.5 * width,
    speedups,
    fmt="none",
    yerr=speedupErrors,
    capsize=3,
    ecolor="k",
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  plt.xlim(xmin=-0.2)
  plt.xticks(X + 0.4, err_fns)
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))

  title = kwargs.pop("title", "Error handler performance for " + job)
  plt.title(title)

  # Add legend *beneath* plot. To do this, we need to pass some
  # extra arguments to plt.savefig(). See:
  #
  # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
  #
  art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
  viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)


def errfn2label(errfn):
  if errfn == "default_fn":
    return r"\textsc{Baseline}"
  elif errfn == "random_fn":
    return r"\textsc{Random}"
  else:
    return r"\textsc{NearestNeighbour}"


def classification(db, output=None, job="xval", **kwargs):
  err_fns = db.err_fns
  base_err_fn = err_fns[0]
  # Get a list of classifiers and result counts.
  query = db.execute(
    "SELECT classifier,Count(*) AS count\n"
    "FROM classification_results\n"
    "WHERE job=? AND err_fn=? AND classifier!='weka.classifiers.rules.ZeroR'\n"
    "GROUP BY classifier",
    (job, base_err_fn),
  )
  results = []

  # Add baseline results.
  baseline = "4x4"
  correct = db.execute(
    "SELECT Count(*) * 1.0 / 3 FROM classification_results "
    "WHERE job=? AND actual=?",
    (job, baseline),
  ).fetchone()[0]
  illegal = 0
  refused = 0
  time = 0
  terr = 0
  speedup = (1, 0)
  perfs = [
    row[1]
    for row in db.execute(
      "SELECT "
      "  DISTINCT runtime_stats.scenario, "
      "  (scenario_stats.oracle_runtime / runtime_stats.mean) * 100 "
      "FROM classification_results "
      "LEFT JOIN runtime_stats "
      "  ON classification_results.scenario=runtime_stats.scenario "
      "LEFT JOIN scenario_stats "
      "  ON classification_results.scenario=scenario_stats.scenario "
      "WHERE job=? and runtime_stats.params=?",
      (job, baseline),
    )
  ]
  perf = (labmath.mean(perfs), labmath.confinterval(perfs, error_only=True))
  results.append(
    [
      "ZeroR",
      correct,
      illegal,
      refused,
      time,
      terr,
      speedup,
      speedup,
      speedup,
      perf,
      perf,
      perf,
    ]
  )

  # Get results
  for classifier, count in query:
    basename = ml.classifier_basename(classifier)
    correct, illegal, refused, time, terr = db.execute(
      "SELECT\n"
      "    (SUM(correct) / CAST(? AS FLOAT)) * 100,\n"
      "    (SUM(illegal) / CAST(? AS FLOAT)) * 100,\n"
      "    (SUM(refused) / CAST(? AS FLOAT)) * 100,\n"
      "    AVG(time) + 2.5,\n"
      "    CONFERROR(time, .95) * 1.5\n"
      "FROM classification_results\n"
      "WHERE job=? AND classifier=? AND err_fn=?",
      (count, count, count, job, classifier, base_err_fn),
    ).fetchone()
    # Get a list of mean speedups for each err_fn.
    speedups = [
      db.execute(
        "SELECT\n"
        "    AVG(speedup),\n"
        "    CONFERROR(speedup, .95)\n"
        "FROM classification_results\n"
        "WHERE job=? AND classifier=? AND err_fn=?",
        (job, classifier, err_fn),
      ).fetchone()
      for err_fn in err_fns
    ]
    # Get a list of mean perfs for each err_fn.
    perfs = [
      db.execute(
        "SELECT\n"
        "    AVG(performance) * 100.0,\n"
        "    CONFERROR(performance, .95) * 100.0\n"
        "FROM classification_results\n"
        "WHERE job=? AND classifier=? AND err_fn=?",
        (job, classifier, err_fn),
      ).fetchone()
      for err_fn in err_fns
    ]

    results.append(
      [basename, correct, illegal, refused, time, terr] + speedups + perfs
    )

  # Zip into lists.
  labels, correct, illegal, refused, time, terr = zip(
    *[
      (
        text.truncate(result[0], 40),
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
      )
      for result in results
    ]
  )

  X = np.arange(len(labels))

  # PLOT TIMES
  width = 0.8
  ax = plt.subplot(4, 1, 1)
  ax.bar(X + 0.1, time, width=width)
  ax.set_xticks(X + 0.4)
  ax.set_xticklabels(labels)
  ax.set_ylim(0, 10)
  ax.set_ylabel("Classification time (ms)")
  # art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -.1), ncol=3)]
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = ax.errorbar(
    X + 0.5, time, fmt="none", yerr=terr, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  # RATIOS
  width = 0.8 / 3
  ax = plt.subplot(4, 1, 2)
  ax.bar(
    X + 0.1,
    illegal,
    width=width,
    color=sns.color_palette("Reds", 1),
    label="Illegal",
  )
  ax.bar(
    X + 0.1 + width,
    refused,
    width=width,
    color=sns.color_palette("Oranges", 1),
    label="Refused",
  )
  ax.bar(
    X + 0.1 + 2 * width,
    correct,
    width=width,
    color=sns.color_palette("Blues", 1),
    label="Accurate",
  )
  ax.set_xticks(X + 0.4)
  ax.set_ylabel("Ratio")
  ax.set_ylim(0, 35)
  ax.set_xticklabels(labels)
  ax.yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]

  # Plot speedups.
  ax = plt.subplot(4, 1, 3)
  width = 0.8 / 3
  colors = sns.color_palette("Greens", len(err_fns))
  for i, err_fn in enumerate(db.err_fns):
    pairs = [result[6 + i] for result in results]
    speedups, yerrs = zip(*pairs)
    ax.bar(
      X + 0.1 + (i * width),
      speedups,
      width=width,
      label=errfn2label(err_fn),
      color=colors[i],
    )

    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _, caps, _ = ax.errorbar(
      X + 0.1 + (i + 0.5) * width,
      speedups,
      fmt="none",
      yerr=yerrs,
      capsize=3,
      ecolor="k",
    )
    for cap in caps:
      cap.set_color("k")
      cap.set_markeredgewidth(1)
  ax.set_xticks(X + 0.4)
  ax.set_xticklabels(labels)
  ax.set_ylim(0, 7)
  ax.set_xticks(X + 0.4, labels)
  ax.set_ylabel("Speedup")
  art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]

  # PERFORMANCE
  colors = sns.color_palette("Blues", len(err_fns))
  width = 0.8 / 3
  ax = plt.subplot(4, 1, 4)
  for i, err_fn in enumerate(db.err_fns):
    pairs = [result[9 + i] for result in results]
    perfs, yerrs = zip(*pairs)
    ax.bar(
      X + 0.1 + (i * width),
      perfs,
      width=width,
      label=errfn2label(err_fn),
      color=colors[i],
    )

    # Plot confidence intervals separately so that we can have
    # full control over formatting.
    _, caps, _ = ax.errorbar(
      X + 0.1 + (i + 0.5) * width,
      perfs,
      fmt="none",
      yerr=yerrs,
      capsize=3,
      ecolor="k",
    )
    for cap in caps:
      cap.set_color("k")
      cap.set_markeredgewidth(1)
  ax.set_xticks(X + 0.4)
  ax.yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  ax.set_xticklabels(labels)
  ax.set_ylim(0, 100)
  ax.set_ylabel("Performance")
  ax.set_xticks(X + 0.4, labels)

  title = kwargs.pop("title", "Classification results for " + job)
  plt.title(title)

  # Add legend *beneath* plot. To do this, we need to pass some
  # extra arguments to plt.savefig(). See:
  #
  # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
  #
  art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
  viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)


def refused_params_by_device(db, output=None, **kwargs):
  data = [
    (fmtdevid(row[0]), round(row[1], 2))
    for row in db.execute(
      "SELECT "
      "    devices.id AS device, "
      "    (Count(*) * 1.0 / ( "
      "        SELECT Count(*) "
      "        FROM runtime_stats "
      "        LEFT JOIN scenarios "
      "          ON runtime_stats.scenario=scenarios.id "
      "        WHERE scenarios.device=devices.id "
      "    )) * 100 AS ratio_refused "
      "FROM refused_params "
      "LEFT JOIN scenarios "
      "  ON refused_params.scenario=scenarios.id "
      "LEFT JOIN devices "
      "  ON scenarios.device=devices.id "
      "GROUP BY devices.id "
      "ORDER BY ratio_refused DESC"
    )
  ]

  labels, Y = zip(*data)
  X = np.arange(len(Y))

  fig, ax = plt.subplots()
  ax.bar(X + 0.1, Y, width=0.8)
  ax.set_xticks(X + 0.5)
  ax.set_xticklabels(labels, rotation=90)
  ax.set_ylabel("Ratio refused (\\%)")

  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))

  for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment("center")

  viz.finalise(output, **kwargs)


def refused_params_by_vendor(db, output=None, **kwargs):
  data = [
    row
    for row in db.execute(
      "SELECT devices.vendor,"
      "    ratio_refused "
      "FROM devices LEFT JOIN ("
      "SELECT "
      "    devices.vendor AS opencl, "
      "    (Count(*) * 1.0 / ( "
      "        SELECT Count(*) "
      "        FROM runtime_stats "
      "        LEFT JOIN scenarios "
      "          ON runtime_stats.scenario=scenarios.id "
      "        LEFT JOIN devices AS dev "
      "          ON scenarios.device=dev.id "
      "        WHERE dev.vendor=devices.vendor "
      "    )) * 100 AS ratio_refused "
      "FROM refused_params "
      "LEFT JOIN scenarios "
      "  ON refused_params.scenario=scenarios.id "
      "LEFT JOIN devices "
      "  ON scenarios.device=devices.id "
      "GROUP BY devices.vendor COLLATE NOCASE )"
      "ON devices.vendor like opencl "
      "GROUP BY devices.vendor COLLATE NOCASE "
      "ORDER BY ratio_refused DESC"
    )
  ]

  labels, Y = zip(*data)
  Y = [0 if not y else y for y in Y]
  X = np.arange(len(Y))

  fig, ax = plt.subplots()
  ax.bar(X + 0.1, Y, width=0.8)
  ax.set_xticks(X + 0.5)
  ax.set_xticklabels(labels, rotation=90)
  ax.set_ylabel("Ratio refused (\\%)")

  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))

  for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment("center")

  viz.finalise(output, **kwargs)
  return data


def refused_param_space(db, output=None, **kwargs):
  space = db.refused_param_space()
  space.heatmap(
    xlabels=False, ylabels=False, cbar=False, output=output, **kwargs
  )


def runtime_regression(db, output=None, job="xval", **kwargs):
  """
  Plot accuracy of a classifier at predicted runtime.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)

  colors = sns.color_palette()
  i, actual = 0, []

  for i, classifier in enumerate(db.regression_classifiers):
    basename = ml.classifier_basename(classifier)
    actual, predicted = zip(
      *sorted(
        [
          row
          for row in db.execute(
            "SELECT\n"
            "    actual,\n"
            "    predicted\n"
            "FROM runtime_regression_results\n"
            "WHERE job=? AND classifier=?",
            (job, classifier),
          )
        ],
        key=lambda x: x[0],
        reverse=True,
      )
    )

    if basename == "ZeroR":
      ax.plot(predicted, label=basename, color=colors[i - 1])
    else:
      ax.scatter(
        np.arange(len(predicted)),
        predicted,
        label=basename,
        color=colors[i - 1],
      )

  ax.plot(actual, label="Actual", color=colors[i])
  ax.set_yscale("log")
  plt.xlim(0, len(actual))
  plt.legend()
  title = kwargs.pop("title", "Runtime regression for " + job)
  plt.title(title)
  plt.xlabel("Test instances (sorted by descending runtime)")
  plt.ylabel("Runtime (ms, log)")
  viz.finalise(output, **kwargs)


def regression_classification(
  db, output=None, job="xval", table="runtime_classification_results", **kwargs
):
  """
  Plot performance of classification using runtime regression.
  """
  jobs = {
    "xval": "10-fold",
    "synthetic_real": "Synthetic",
    "arch": "Device",
    "kern": "Kernel",
    "data": "Dataset",
  }

  results = []
  for job in jobs:
    speedup, serr, perf, perr, time, terr, correct = db.execute(
      "SELECT "
      "  AVG(speedup), CONFERROR(speedup, .95), "
      "  AVG(performance) * 100, CONFERROR(performance, .95) * 100, "
      "  AVG(time) + 2.5, CONFERROR(time, .95), "
      "  AVG(correct) * 100 "
      "FROM {} WHERE job=?".format(table),
      (job,),
    ).fetchone()
    results.append([job, speedup, serr, perf, perr, time, terr, correct])

  # Zip into lists.
  labels, speedup, serr, perf, perr, time, terr, correct = zip(*results)
  labels = [jobs[x] for x in jobs]

  # Add averages.
  labels.append(r"\textbf{Average}")
  speedup += (labmath.mean(speedup),)
  serr += (labmath.mean(serr),)
  perf += (labmath.mean(perf),)
  perr += (labmath.mean(perr),)
  time += (labmath.mean(time),)
  terr += (labmath.mean(terr),)
  correct += (labmath.mean(correct),)

  X = np.arange(len(labels))

  width = 0.8

  # PLOT TIMES
  ax = plt.subplot(4, 1, 1)
  ax.bar(X + 0.1, time, width=width)
  ax.set_xticks(X + 0.5)
  ax.set_ylim(0, 150)
  ax.set_xticklabels(labels, rotation="vertical")
  ax.set_ylabel("Classification time (ms)")
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = ax.errorbar(
    X + 0.5, time, fmt="none", yerr=terr, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  # SPEEDUPS
  ax = plt.subplot(4, 1, 3)
  ax.bar(X + 0.1, speedup, width=width, color=sns.color_palette("Greens"))
  ax.set_xticks(X + 0.5)
  ax.set_ylim(0, 7)
  ax.set_xticklabels(labels, rotation="vertical")
  ax.set_ylabel("Speedup")
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = ax.errorbar(
    X + 0.5, speedup, fmt="none", yerr=serr, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  # PERFORMANCE
  ax = plt.subplot(4, 1, 4)
  ax.bar(X + 0.1, perf, width=width, color=sns.color_palette("Blues"))
  ax.set_xticks(X + 0.5)
  ax.set_xticklabels(labels, rotation="vertical")
  ax.set_ylabel("Performance")
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  ax.set_ylim(0, 100)
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = ax.errorbar(
    X + 0.5, perf, fmt="none", yerr=perr, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  # ACCURACY
  ax = plt.subplot(4, 1, 2)
  ax.bar(X + 0.1, correct, width=width, color=sns.color_palette("Reds"))
  ax.set_xticks(X + 0.5)
  ax.set_xticklabels(labels, rotation="vertical")
  ax.set_ylabel("Accuracy")
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))
  ax.set_ylim(0, 12)

  viz.finalise(output, **kwargs)


def speedup_regression(db, output=None, job="xval", **kwargs):
  """
  Plot accuracy of a classifier at predicted runtime.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)

  colors = sns.color_palette()
  for i, classifier in enumerate(db.regression_classifiers):
    basename = ml.classifier_basename(classifier)
    actual, predicted = zip(
      *sorted(
        [
          row
          for row in db.execute(
            "SELECT\n"
            "    actual,\n"
            "    predicted\n"
            "FROM speedup_regression_results\n"
            "WHERE job=? AND classifier=?",
            (job, classifier),
          )
        ],
        key=lambda x: x[0],
        reverse=True,
      )
    )

    if basename == "ZeroR":
      ax.plot(predicted, label=basename, color=colors[i - 1])
    else:
      ax.scatter(
        np.arange(len(predicted)),
        predicted,
        label=basename,
        color=colors[i - 1],
      )

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
    "WHERE job=? GROUP BY classifier",
    (job,),
  )
  results = []
  for classifier, count in query:
    basename = ml.classifier_basename(classifier)
    correct = db.execute(
      "SELECT\n"
      "    (SUM(correct) / CAST(? AS FLOAT)) * 100\n"
      "FROM speedup_classification_results\n"
      "WHERE job=? AND classifier=?",
      (count, job, classifier),
    ).fetchone()[0]
    # Get a list of mean speedups for each err_fn.
    speedups = [
      row
      for row in db.execute(
        "SELECT\n"
        "    AVG(speedup) * 100,\n"
        "    CONFERROR(speedup, .95) * 100,\n"
        "    AVG(performance) * 100,\n"
        "    CONFERROR(performance, .95) * 100\n"
        "FROM speedup_classification_results\n"
        "WHERE job=? AND classifier=?",
        (job, classifier),
      ).fetchone()
    ]

    results.append([basename, correct] + speedups)

  # Zip into lists.
  labels, correct, speedups, yerrs, perfs, perf_yerrs = zip(*results)

  X = np.arange(len(labels))
  # Bar width.
  width = 0.8 / (len(results[0]) - 1)

  plt.bar(
    X + width,
    correct,
    width=width,
    color=sns.color_palette("Blues", 1),
    label="Accuracy",
  )
  plt.bar(
    X + 2 * width,
    speedups,
    width=width,
    color=sns.color_palette("Greens", 1),
    label="Speedup",
  )
  plt.bar(
    X + 3 * width,
    perfs,
    width=width,
    color=sns.color_palette("Oranges", 1),
    label="Performance",
  )
  # Plot confidence intervals separately so that we can have
  # full control over formatting.
  _, caps, _ = plt.errorbar(
    X + 2.5 * width, speedups, fmt="none", yerr=yerrs, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)
  _, caps, _ = plt.errorbar(
    X + 3.5 * width, perfs, fmt="none", yerr=perf_yerrs, capsize=3, ecolor="k"
  )
  for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(1)

  plt.xlim(xmin=-0.2)
  plt.xticks(X + 0.4, labels)
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d\\%%"))

  title = kwargs.pop(
    "title", "Classification results for " + job + " using speedup regression"
  )
  plt.title(title)

  # Add legend *beneath* plot. To do this, we need to pass some
  # extra arguments to plt.savefig(). See:
  #
  # http://jb-blog.readthedocs.org/en/latest/posts/12-matplotlib-legend-outdide-plot.html
  #
  art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)]
  viz.finalise(output, additional_artists=art, bbox_inches="tight", **kwargs)
