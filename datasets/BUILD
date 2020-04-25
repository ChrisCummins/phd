# The package contains datasets for use in experiments.

# A toy dataset used as an example for ID3 classification algorithm.
#
# Taken from: https://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
#
# The CSV file contains a header row, followed by fourteen entries with the
# format:
#
#     Day: D[1-14]
#     Outlook: {Rain,Overcast,Sunny}
#     Temperature: {Cool,Mild,Hot}
#     Humidity: {Normal,High}
#     Wind: {Weak,Strong}
#     Baseball: {No,Yes}
#
# The goal is to predict the value for the Baseball column, given the four
# features.
filegroup(
    name = "baseball_csv",
    srcs = ["baseball.csv"],
    visibility = ["//visibility:public"],
)
