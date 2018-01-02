"""
me - Aggregate health and time tracking data.
"""
import datetime
import os


def daterange(start_date, end_date, reverse=False):
    """ returns an iterator over the specified date range """
    for n in range(int((end_date - start_date).days)):
        if reverse:
            yield end_date - datetime.timedelta(n)
        else:
            yield start_date + datetime.timedelta(n)


def mkdir(path):
    """ make directory if it does not already exist """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
