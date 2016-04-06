#!/usr/bin/env python
#
# Exploratory analysis of preprocessed dataset.
#
# TODO:
#
# High level summary:
#   Number of files
#   Number of repositories
#   Total line count
#
# Graphs:
#   Number of OpenCL repos over time
#   Distribution of stars and OpenCL file counts
#   Distribution of repo star counts
#   Distribution of file star counts
#   Distribution of repo forks
#   Distribution of times since last changed
#   Distribution of file sizes
#   Distribution of files per repo

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
