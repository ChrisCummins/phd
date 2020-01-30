// Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "datasets/me_db/providers/health_kit/health_kit_lib.h"

#include "datasets/me_db/me.pb.h"

#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

namespace me {
namespace {

TEST(ParseHealthKitDatetimeOrDie, UnixEpoch) {
  EXPECT_EQ(ParseHealthKitDatetimeOrDie("1970-01-01 00:00:00 +0000"), 0);
}

TEST(ParseHealthKitDatetimeOrDie, InvalidDate) {
  ASSERT_DEATH(ParseHealthKitDatetimeOrDie("This will crash"),
               "Failed to parse HealthKit datetime 'This will crash'");
}

TEST(ParseHealthKitDatetimeOrDie, DateOnly) {
  // Testing with a known date.
  EXPECT_EQ(ParseHealthKitDatetimeOrDie("2017-01-01 00:00:00 +0000"),
            1483228800000);
}

TEST(ParseHealthKitDatetimeOrDie, DateAndHour) {
  // Testing with a known date.
  EXPECT_EQ(ParseHealthKitDatetimeOrDie("2017-01-01 01:00:00 +0000"),
            1483232400000);
}

TEST(ParseHealthKitDatetimeOrDie, DateAndHourAndMinute) {
  // Testing with a known date.
  EXPECT_EQ(ParseHealthKitDatetimeOrDie("2017-01-01 01:01:00 +0000"),
            1483232460000);
}

TEST(ParseHealthKitDatetimeOrDie, DateAndHourAndMinuteAndSecond) {
  // Testing with a known date.
  EXPECT_EQ(ParseHealthKitDatetimeOrDie("2017-01-01 01:01:01 +0000"),
            1483232461000);
}

TEST(ParseHealthKitDatetimeOrDie, TimeZone) {
  EXPECT_EQ(
      ParseHealthKitDatetimeOrDie("2017-01-01 01:01:01 +0000"),
      ParseHealthKitDatetimeOrDie("2017-01-01 01:01:01 +0100") + 3600 * 1000);
}

TEST(ParseIntOrDie, EmptyString) {
  ASSERT_DEATH(ParseIntOrDie("This will crash"),
               "Failed to parse integer 'This will crash'");
}

TEST(ParseIntOrDie, One) { EXPECT_EQ(ParseIntOrDie("1"), 1); }

TEST(ParseIntOrDie, Negative) { EXPECT_EQ(ParseIntOrDie("-123"), -123); }

TEST(ParseIntOrDie, Float) {
  ASSERT_DEATH(ParseIntOrDie("1.53"), "Failed to parse integer '1.53'");
}

TEST(ParseDoubleOrDie, EmptyString) {
  ASSERT_DEATH(ParseDoubleOrDie("This will crash"),
               "Failed to parse double 'This will crash'");
}

TEST(ParseDoubleOrDie, One) { EXPECT_EQ(ParseDoubleOrDie("1"), 1); }

TEST(ParseDoubleOrDie, Negative) { EXPECT_EQ(ParseDoubleOrDie("-123"), -123); }

TEST(ParseDoubleOrDie, Double) { EXPECT_EQ(ParseDoubleOrDie("1.53"), 1.53); }

TEST(SetAttributeIfMatch, NoMatch) {
  boost::property_tree::ptree ptree;
  ptree.put("KEY", "VALUE");

  string s;
  SetAttributeIfMatch(ptree.front(), "not a match", &s);
  EXPECT_TRUE(s.empty());
}

TEST(SetAttributeIfMatch, Match) {
  boost::property_tree::ptree ptree;
  ptree.put("KEY", "VALUE");

  string s;
  SetAttributeIfMatch(ptree.front(), "KEY", &s);
  EXPECT_EQ(s, "VALUE");
}

TEST(HealthKitRecordImporter, DebugString) {
  HealthKitRecordImporter importer("type", "unit", "value", "source",
                                   "start_date", "end_date");
  EXPECT_EQ(importer.DebugString(),
            "type=type\nvalue=value\nunit=unit\nsource=source\n"
            "start_date=start_date\nend_date=end_date");
}

TEST(AddMeasurementsOrDie, InvalidValues) {
  SeriesCollection series_collection;
  absl::flat_hash_map<string, Series*> type_to_series_map;
  HealthKitRecordImporter importer("type", "unit", "value", "source",
                                   "start_date", "end_date");
  ASSERT_DEATH(
      importer.AddMeasurementsOrDie(&series_collection, &type_to_series_map),
      "Unhandled type 'type' for HealthKit record");
}

TEST(AddMeasurementsOrDie, SeriesIsAddedToTypeMap) {
  SeriesCollection series_collection;
  absl::flat_hash_map<string, Series*> type_to_series_map;

  HealthKitRecordImporter importer("HKQuantityTypeIdentifierDietaryWater", "mL",
                                   "125", "source", "1970-01-01 00:00:01 +0000",
                                   "end_date");
  importer.AddMeasurementsOrDie(&series_collection, &type_to_series_map);

  ASSERT_EQ(type_to_series_map.size(), 1);

  auto it = type_to_series_map.find("HKQuantityTypeIdentifierDietaryWater");
  ASSERT_NE(it, type_to_series_map.end());
  EXPECT_EQ(it->second->name(), "WaterConsumed");
}

TEST(AddMeasurementsOrDie, HKQuantityTypeIdentifierDietaryWater) {
  SeriesCollection series_collection;
  absl::flat_hash_map<string, Series*> type_to_series_map;

  HealthKitRecordImporter importer("HKQuantityTypeIdentifierDietaryWater", "mL",
                                   "125", "source", "1970-01-01 00:00:01 +0000",
                                   "end_date");
  importer.AddMeasurementsOrDie(&series_collection, &type_to_series_map);

  ASSERT_EQ(series_collection.series_size(), 1);
  Series series = series_collection.series(0);
  EXPECT_EQ(series.family(), "Diet");
  EXPECT_EQ(series.name(), "WaterConsumed");
  EXPECT_EQ(series.unit(), "milliliters");

  ASSERT_EQ(series.measurement_size(), 1);
  Measurement measurement = series_collection.series(0).measurement(0);
  EXPECT_EQ(measurement.ms_since_unix_epoch(), 1000);
  EXPECT_EQ(measurement.value(), 125);
  EXPECT_EQ(measurement.group(), "default");
  EXPECT_EQ(measurement.source(), "HealthKit:Source");
}

TEST(AddMeasurementsOrDie, HKQuantityTypeIdentifierBodyMassIndex) {
  SeriesCollection series_collection;
  absl::flat_hash_map<string, Series*> type_to_series_map;

  HealthKitRecordImporter importer("HKQuantityTypeIdentifierBodyMassIndex",
                                   "count", "23.5", "foo",
                                   "1970-01-01 00:00:01 +0000", "end_date");
  importer.AddMeasurementsOrDie(&series_collection, &type_to_series_map);

  ASSERT_EQ(series_collection.series_size(), 1);
  Series series = series_collection.series(0);
  EXPECT_EQ(series.family(), "BodyMeasurements");
  EXPECT_EQ(series.name(), "BodyMassIndex");
  EXPECT_EQ(series.unit(), "body_mass_index_millis");

  ASSERT_EQ(series.measurement_size(), 1);
  Measurement measurement = series_collection.series(0).measurement(0);
  EXPECT_EQ(measurement.ms_since_unix_epoch(), 1000);
  EXPECT_EQ(measurement.value(), 23500000);
  EXPECT_EQ(measurement.group(), "default");
  EXPECT_EQ(measurement.source(), "HealthKit:Foo");
}

}  // namespace
}  // namespace me

TEST_MAIN();
