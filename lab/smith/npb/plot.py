#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import sys

import csv

import labm8
from labm8 import math as labmath

def get_data_from_file(path):
    dataset=[]
    cpu=[]
    gpu=[]

    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            benchmark = row[0]
            dataset.append(row[1])
            cpu.append(float(row[2]))
            gpu.append(float(row[3]))
    return benchmark, dataset, cpu, gpu

def condense_data(dataset, cpu, gpu):
    uniq = dict((x,([],[])) for x in set(dataset))

    for i in range(len(dataset)):
        d,c,g = dataset[i], cpu[i], gpu[i]
        uniq[d][0].append(c)
        uniq[d][1].append(g)

    for key in uniq:
        d = uniq[key]
        uniq[key] = (
            (labmath.mean(d[0]),
             labmath.confinterval(d[0])[1] - labmath.mean(d[0])),
            (labmath.mean(d[1]),
             labmath.confinterval(d[1])[1] - labmath.mean(d[1])))

    return uniq

def plottable(data):
    labels = ['S', 'W', 'A', 'B', 'C', 'D', 'E']

    classes, cpu, cpu_err, gpu, gpu_err = [], [], [], [], []

    for c in labels:
        if c not in data:
            continue

        classes.append(c)
        d = data[c]
        cpu.append(d[0][0])
        cpu_err.append(d[0][1])

        gpu.append(d[1][0])
        gpu_err.append(d[1][1])

    if len(classes) != len(data):
        print("YALL FUKD UP!")
    return classes,cpu,cpu_err,gpu,gpu_err


path = sys.argv[1]
if len(sys.argv) == 3:
    dst = sys.argv[2]
else:
    dst = None
benchmark, dataset, cpu, gpu = get_data_from_file(path)

data = condense_data(dataset, cpu, gpu)
labels, cpu, cpu_err, gpu, gpu_err = plottable(data)


N = len(labels)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, cpu, width, color='r', yerr=cpu_err)
rects2 = ax.bar(ind + width, gpu, width, color='y', yerr=gpu_err)

# add some text for labels, title and axes ticks
ax.set_ylabel('Runtime (s)')
ax.set_title('npb-3.3 benchmark {}'.format(benchmark))
ax.set_xticks(ind + width)
ax.set_xticklabels(labels)

ax.legend((rects1[0], rects2[0]), ('CPU', 'GPU'))

def autolabel(rects, speedups):
    # attach some text labels
    for rect, speedup in zip(rects, speedups):
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.07 * rect.get_height(),
                '{:.2f}'.format(speedup),
                ha='center', va='bottom')


speedups = [c / g for c,g in zip(cpu,gpu)]
autolabel(rects2, speedups)

if dst:
    plt.savefig(dst)
else:
    plt.show()
