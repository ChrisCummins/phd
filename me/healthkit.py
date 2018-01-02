import csv
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


def create_sum_csv(records, column_name, unit, outpath):

    def _attr_filter(record):
        assert(record['unit'].value == unit)

    with open(outpath, "w") as outfile:
        logging.debug(f"Creating CSV file {outfile.name}")
        writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        writer.writerow(("Date", column_name))
        rows = sum_values_by_day(records, filter_fn=_attr_filter)
        for row in rows:
            writer.writerow(row)

    nrows = len(rows)
    logging.info(f"Exported {nrows} records to \"{outfile.name}\"")


def create_avg_csv(records, column_name, unit, outpath, min_max: bool=False):

    def _attr_filter(record):
        assert(record['unit'].value == unit)

    with open(outpath, "w") as outfile:
        logging.debug(f"Creating CSV file {outfile.name}")
        writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        if min_max:
            writer.writerow(("Date", f"Min {column_name}", f"Avg {column_name}",
                             f"Max {column_name}"))
        else:
            writer.writerow(("Date", column_name))
        rows = avg_values_by_day(records, filter_fn=_attr_filter, min_max=min_max)
        for row in rows:
            writer.writerow(row)

    nrows = len(rows)
    logging.info(f"Exported {nrows} records to \"{outfile.name}\"")


def _process_records_generic(typename, records, outdir):
    # build a list of attributes names (columns)
    attributes = get_all_atributes(records, ["type"])

    # Create CSV file
    with open(f"{outdir}/{typename}.csv", "w") as outfile:
        logging.debug(f"Creating CSV file {outfile.name}")
        writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(attributes)

        # Write rows
        for record in records:
            row = []
            for attr in attributes:
                try:
                    row.append(record[attr].value)
                except:
                    row.append('')
            writer.writerow(row)

        nrows = len(records)
        logging.info(f"Exported {nrows} records for \"{typename}\"")


def _process_step_count(typename, records, outdir):
    create_sum_csv(records, "Step Count", "count", f"{outdir}/Step Count.csv")


def _process_body_mass(typename, records, outdir):
    create_sum_csv(records, "Weight (kg)", "kg", f"{outdir}/Weight.csv")


def _process_dietary_energy_consumed(typename, records, outdir):
    create_sum_csv(records, "Calories Consumed (kcal)", "kcal", f"{outdir}/Calories Consumed.csv")


def _process_active_energy_burned(typename, records, outdir):
    create_sum_csv(records, "Calories Burned (kcal)", "kcal", f"{outdir}/Calories Burned.csv")


def _process_dietary_water(typename, records, outdir):
    create_sum_csv(records, "Water (mL)", "mL", f"{outdir}/Water.csv")


def _process_dietary_caffeine(typename, records, outdir):
    create_sum_csv(records, "Caffeine (mg)", "mg", f"{outdir}/Caffeine.csv")


def _process_body_mass_index(typename, records, outdir):
    create_avg_csv(records, "Body Mass Index", "count", f"{outdir}/BMI.csv")


def _process_distance_walking_running(typename, records, outdir):
    create_sum_csv(records, "Distance Walking + Running (km)", "km",
                   f"{outdir}/Distance.csv")


def _process_heart_rate(typename, records, outdir):
    create_avg_csv(records, "Heart Rate (bmp)", "count/min",
                   f"{outdir}/Heart Rate.csv", min_max=True)


def _process_height(typename, records, outdir):
    create_avg_csv(records, "Height (cm)", "cm", f"{outdir}/Height.csv")


def _process_dietary_fat_total(typename, records, outdir):
    create_sum_csv(records, "Total Fat (g)", "g",
                   f"{outdir}/Total Fat.csv")


def _process_dietary_fat_saturated(typename, records, outdir):
    create_sum_csv(records, "Saturated Fat (g)", "g",
                   f"{outdir}/Sat Fat.csv")


def _process_dietary_cholesterol(typename, records, outdir):
    create_sum_csv(records, "Cholesterol (mg)", "mg",
                   f"{outdir}/Cholesterol.csv")


def _process_dietary_sodium(typename, records, outdir):
    create_sum_csv(records, "Sodium (mg)", "mg",
                   f"{outdir}/Sodium.csv")


def _process_dietary_carbohydrates(typename, records, outdir):
    create_sum_csv(records, "Carbohydrates (g)", "g",
                   f"{outdir}/Carbohydrates.csv")


def _process_dietary_fiber(typename, records, outdir):
    create_sum_csv(records, "Fiber (g)", "g",
                   f"{outdir}/Fiber.csv")


def _process_dietary_sugar(typename, records, outdir):
    create_sum_csv(records, "Sugar (g)", "g",
                   f"{outdir}/Sugar.csv")


def _process_dietary_protein(typename, records, outdir):
    create_sum_csv(records, "Protein (g)", "g",
                   f"{outdir}/Protein.csv")


def _process_dietary_potassium(typename, records, outdir):
    create_sum_csv(records, "Potassium (mg)", "mg",
                   f"{outdir}/Potassium.csv")


def process_records(typename, records, outdir):
    handler = {
        "Active Energy Burned": _process_active_energy_burned,
        "Body Mass Index": _process_body_mass_index,
        "Body Mass": _process_body_mass,
        "Dietary Caffeine": _process_dietary_caffeine,
        "Dietary Carbohydrates": _process_dietary_carbohydrates,
        "Dietary Cholesterol": _process_dietary_cholesterol,
        "Dietary Energy Consumed": _process_dietary_energy_consumed,
        "Dietary Fat Saturated": _process_dietary_fat_saturated,
        "Dietary Fat Total": _process_dietary_fat_total,
        "Dietary Fiber": _process_dietary_fiber,
        "Dietary Potassium": _process_dietary_potassium,
        "Dietary Protein": _process_dietary_protein,
        "Dietary Sodium": _process_dietary_sodium,
        "Dietary Sugar": _process_dietary_sugar,
        "Dietary Water": _process_dietary_water,
        "Distance Walking Running": _process_distance_walking_running,
        "Heart Rate": _process_heart_rate,
        "Height": _process_height,
        "Step Count": _process_step_count,
    }.get(typename, _process_records_generic)

    return handler(typename, records, outdir)


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
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    with TemporaryDirectory(prefix="me.csv.") as unzipdir:
        logging.debug(f"Unpacking healthkit archive to {unzipdir}")
        archive = ZipFile(infile.name)
        archive.extract("apple_health_export/export.xml", path=unzipdir)
        with open(f"{unzipdir}/apple_health_export/export.xml") as infile:
            process_file(infile, outdir)
