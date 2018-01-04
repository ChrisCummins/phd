import datetime
import logging
import os
import re

from argparse import ArgumentParser, FileType
from collections import defaultdict
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from xml.dom import minidom

import me


def get_all_atributes(records, excludes=[]):
    attributes = set()
    for record in records:
        for attr in record.keys():
            attributes.add(attr)
    attributes = sorted([x for x in attributes if x not in excludes])
    return attributes


def parse_datetime(string):
    return datetime.datetime.strptime(str(string), "%Y-%m-%d %H:%M:%S %z")


def parse_date(string):
    return parse_datetime(string).date()


def sum_values_by_day(records, value_attr: str='value',
                      date_attr: str='endDate', filter_fn=None):
    rows = []
    for r in records:
        if filter_fn and not filter_fn(r):
            rows.append((parse_date(r[date_attr].value),
                         float(r[value_attr].value)))
    rows = sorted(rows, key=lambda x: x[0])

    rows2 = []
    last_date = None
    aggr = 0
    for row in rows:
        date, value = row
        if date != last_date:
            if last_date:
                rows2.append((last_date, aggr))
            last_date = date
            aggr = value
        else:
            aggr += value
    rows2.append((last_date, aggr))

    return rows2


def sum_durations_by_day(records, filter_fn=None):
    rows = []
    for r in records:
        if filter_fn and not filter_fn(r):
            rows.append((parse_date(r['startDate'].value),
                         parse_datetime(r['endDate']) - parse_datetime(r['startDate'])))
    rows = sorted(rows, key=lambda x: x[0])

    rows2 = []
    last_date = None
    aggr = 0
    for row in rows:
        date, value = row
        if date != last_date:
            if last_date:
                rows2.append((last_date, aggr))
            last_date = date
            aggr = value
        else:
            aggr += value
    rows2.append((last_date, aggr))

    return rows2


def avg_values_by_day(records, value_attr: str='value',
                      date_attr: str='endDate', filter_fn=None,
                      min_max: bool=False):
    rows = []
    for r in records:
        if filter_fn and not filter_fn(r):
            rows.append((parse_date(r[date_attr].value),
                         float(r[value_attr].value)))
    rows = sorted(rows, key=lambda x: x[0])

    rows2 = []
    last_date = None
    aggr = []
    for row in rows:
        date, value = row
        if date != last_date:
            if last_date:
                if min_max:
                    rows2.append((last_date, min(aggr), sum(aggr) / len(aggr), max(aggr)))
                else:
                    rows2.append((last_date, sum(aggr) / len(aggr)))
            last_date = date
            aggr = [value]
        else:
            aggr.append(value)
    if min_max:
        rows2.append((last_date, min(aggr), sum(aggr) / len(aggr), max(aggr)))
    else:
        rows2.append((last_date, sum(aggr) / len(aggr)))

    return rows2


def create_sum_csv(records, name, unit, outpath):

    def _attr_filter(record):
        assert(record['unit'].value == unit)

    header = ("Date", name)
    rows = sum_values_by_day(records, filter_fn=_attr_filter)
    me.create_csv([header] + rows, outpath)


def create_avg_csv(records, name, unit, outpath, min_max: bool=False):

    def _attr_filter(record):
        assert(record['unit'].value == unit)

    if min_max:
        header = ("Date", f"Min {name}", f"Avg {name}", f"Max {name}")
    else:
        header = ("Date", name)
    rows = avg_values_by_day(records, filter_fn=_attr_filter, min_max=min_max)
    me.create_csv([header] + rows, outpath)


def create_stand_hour_csv(records, outpath):
    records = sorted(records, key=lambda r: r['startDate'].value)

    rows = [("Date", "Stand Hours", "Idle Hours")]
    last_date = None
    counts = [0, 0]
    for r in records:
        date = parse_date(r['startDate'].value)

        if date != last_date:
            if last_date:
                rows.append([last_date] + counts)
            last_date = date
            counts = [1, 1]
        else:
            if r['value'].value == "HKCategoryValueAppleStandHourStood":
                counts[0] += 1
            elif r['value'].value == "HKCategoryValueAppleStandHourIdle":
                counts[1] += 1
            else:
                raise ValueError("unrecognized value " + str(r['value'].value))

    rows.append([last_date] + counts)
    me.create_csv(rows, outpath)


def _process_records_generic(typename, records, outdir):
    # build a list of attributes names (columns)
    attributes = get_all_atributes(records, ["type"])

    outpath = f"{outdir}/{typename}.csv"

    header = attributes
    rows = []
    for record in records:
        row = []
        for attr in attributes:
            try:
                row.append(record[attr].value)
            except:
                row.append('')
        rows.append(row)

    me.create_csv([header] + rows, outpath)


def process_records(typename, records, outdir):
    handler = {
        "Active Energy Burned": {
            "fn": create_sum_csv,
            "name": "Active Energy (kcal)",
            "unit": "kcal",
            "dest": "Active Energy",
        },
        "Apple Exercise Time": {
            "fn": create_sum_csv,
            "name": "Exercise Time",
            "unit": "min",
            "dest": "Exercise Time",
        },
        "Apple Stand Hour": {
            "fn": create_stand_hour_csv,
            "dest": "Stand Hours"
        },
        "Basal Energy Burned": {
            "fn": create_sum_csv,
            "name": "Resting Energy (kcal)",
            "unit": "kcal",
            "dest": "Resting Energy",
        },
        "Body Mass Index": {
            "fn": create_avg_csv,
            "name": "Body Mass Index",
            "unit": "count",
            "dest": "BMI",
        },
        "Body Mass": {
            "fn": create_avg_csv,
            "name": "Weight (kg)",
            "unit": "kg",
            "dest": "Weight",
        },
        "Dietary Caffeine": {
            "fn": create_sum_csv,
            "name": "Caffeine (mg)",
            "unit": "mg",
            "dest": "Caffeine",
        },
        "Dietary Carbohydrates": {
            "fn": create_sum_csv,
            "name": "Carbohydrates (g)",
            "unit": "g",
            "dest": "Carbohydrates",
        },
        "Dietary Cholesterol": {
            "fn": create_sum_csv,
            "name": "Cholesterol (mg)",
            "unit": "mg",
            "dest": "Cholesterol",
        },
        "Dietary Energy Consumed": {
            "fn": create_sum_csv,
            "name": "Calories Consumed (kcal)",
            "unit": "kcal",
            "dest": "Calories Consumed",
        },
        "Dietary Fat Saturated": {
            "fn": create_sum_csv,
            "name": "Saturated Fat (g)",
            "unit": "g",
            "dest": "Sat Fat",
        },
        "Dietary Fat Total": {
            "fn": create_sum_csv,
            "name": "Total Fat (g)",
            "unit": "g",
            "dest": "Total Fat",
        },
        "Dietary Fiber": {
            "fn": create_sum_csv,
            "name": "Fiber (g)",
            "unit": "g",
            "dest": "Fiber",
        },
        "Dietary Potassium": {
            "fn": create_sum_csv,
            "name": "Potassium (mg)",
            "unit": "mg",
            "dest": "Potassium",
        },
        "Dietary Protein": {
            "fn": create_sum_csv,
            "name": "Protein (g)",
            "unit": "g",
            "dest": "Protein",
        },
        "Dietary Sodium": {
            "fn": create_sum_csv,
            "name": "Sodium (mg)",
            "unit": "mg",
            "dest": "Sodium",
        },
        "Dietary Sugar": {
            "fn": create_sum_csv,
            "name": "Sugar (g)",
            "unit": "g",
            "dest": "Sugar",
        },
        "Dietary Water": {
            "fn": create_sum_csv,
            "name": "Water (mL)",
            "unit": "mL",
            "dest": "Water",
        },
        "Distance Cycling": {
            "fn": create_sum_csv,
            "name": "Distance Cycling (km)",
            "unit": "km",
            "dest": "Distance Cycling",
        },
        "Distance Walking Running": {
            "fn": create_sum_csv,
            "name": "Distance on Foot (km)",
            "unit": "km",
            "dest": "Distance on Foot",
        },
        "Flights Climbed": {
            "fn": create_sum_csv,
            "name": "Flights Climbed",
            "unit": "count",
            "dest": "Flights Climbed",
        },
        "Heart Rate": {
            "fn": create_avg_csv,
            "name": "Heart Rate (bmp)",
            "unit": "count/min",
            "dest": "Heart Rate.csv",
            "min_max": True
        },
        "Height": {
            "fn": create_avg_csv,
            "name": "Height (cm)",
            "unit": "cm",
            "dest": "Height"
        },
        "Resting Heart Rate": {
            "fn": create_avg_csv,
            "name": "Resting Heart Rate (bmp)",
            "unit": "count/min",
            "dest": "Resting Heart Rate.csv",
            "min_max": True
        },
        "Step Count": {
            "fn": create_sum_csv,
            "name": "Step Count",
            "unit": "count",
            "dest": "Step Count"
        },
        "Walking Heart Rate Average": {
            "fn": create_avg_csv,
            "name": "Walking Heart Rate (bmp)",
            "unit": "count/min",
            "dest": "Walking Heart Rate.csv",
            "min_max": True
        },
    }.get(typename)

    if handler:
        csv_fn = handler.pop("fn")
        outpath = f"{outdir}/" + handler.pop("dest") + ".csv"
        csv_fn(records=records, outpath=outpath, **handler)
    else:
        return _process_records_generic(typename, records, outdir)


def process_file(infile, outdir):
    logging.debug(f"Parsing export.xml")
    xmldoc = minidom.parse(infile.name)
    recordlist = xmldoc.getElementsByTagName('Record')

    data = defaultdict(list)
    for s in recordlist:
        typename = s.attributes['type'].value
        # Strip the HealthKit prefix from the type name:
        typename = typename[len("HKQuantityTypeIdentifier"):]
        # Split the CamelCase name into separate words:
        typename = " ".join(re.findall('[A-Z][^A-Z]*', typename))

        data[typename].append(s.attributes)

    for typename in data:
        process_records(typename, data[typename], outdir)


def process_archive(infile, outdir):
    me.mkdir(outdir)

    with TemporaryDirectory(prefix="me.csv.") as unzipdir:
        logging.debug(f"Unpacking healthkit archive to {unzipdir}")
        archive = ZipFile(infile.name)
        archive.extract("apple_health_export/export.xml", path=unzipdir)
        with open(f"{unzipdir}/apple_health_export/export.xml") as infile:
            process_file(infile, outdir)
