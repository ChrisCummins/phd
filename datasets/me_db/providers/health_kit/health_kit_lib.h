// Library for processing HealthKit export.xml files.
//
// See //datasets/me_db/providers/health_kit:README.md for an overview of the
// schema that is processed, and the measurements produced.
//
// Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
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
#pragma once

#include "datasets/me_db/me.pb.h"

#include "phd/string.h"

#include "absl/container/flat_hash_map.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include "boost/filesystem.hpp"
#include "boost/tokenizer.hpp"

namespace me {

template <typename K, typename V>
V FindOrAdd(absl::flat_hash_map<K, V>* map, const K& key,
            std::function<V(const K&)> add_callback) {
  auto it = map->find(key);

  if (it == map->end()) {
    return add_callback(key);
  } else {
    return it->second;
  }
}

// Parses a date string and return milliseconds since the Unix epoch. If the
// string cannot be parsed, crash fatally. Date string must be in the format
// used by HealthKit: %Y-%m-%d %H:%M:%S %z.
int64_t ParseHealthKitDatetimeOrDie(const string& date);

// Parse an integer from a string or crash fatally.
int64_t ParseIntOrDie(const string& integer_string);

// Parse a double from a string or crash fatally.
double ParseDoubleOrDie(const string& double_string);

// If the given attribute's name matches attribute_name, set attribute_value and
// return true. If the names do not match, attribute_value is not set, and
// returns false.
bool SetAttributeIfMatch(
    const boost::property_tree::ptree::value_type& attribute,
    const string& attribute_name, string* attribute_value);

// A class for processing the Record elements in HealthKit XML files.
//
// Parses the XML attributes, and creates Measurement protos.
//
// Schema for a Record:
//
//   Name:         Attributes:       Example value:
//   type          CDATA #REQUIRED   "HKQuantityTypeIdentifierDietaryWater"
//   unit          CDATA #IMPLIED    "mL"
//   value         CDATA #IMPLIED    "125"
//   sourceName    CDATA #REQUIRED   "Workflow"
//   sourceVersion CDATA #IMPLIED    "494"
//   device        CDATA #IMPLIED
//   creationDate  CDATA #IMPLIED    "2017-12-28 17:25:33 +0100"
//   startDate     CDATA #REQUIRED   "2017-12-28 17:25:33 +0100"
//   endDate       CDATA #REQUIRED   "2017-12-28 17:25:33 +0100"
//
class HealthKitRecordImporter {
 public:
  HealthKitRecordImporter() : new_series_(false) {}

  // Constructor that explicitly sets all fields that would otherwise be set by
  // InitFromRecordOrDie(). Used for testing.
  HealthKitRecordImporter(const string& type, const string& unit,
                          const string& value, const string& sourceName,
                          const string& startDate, const string& endDate)
      : type_(type),
        unit_(unit),
        value_(value),
        sourceName_(sourceName),
        startDate_(startDate),
        endDate_(endDate) {}

  // Initialize the member variables by parsing the XML Record attributes.
  // Instances of this class can be reused by calling this method with different
  // record arguments.
  void InitFromRecordOrDie(const boost::property_tree::ptree& record);

  // Process an XML Record element. Parses the attributes, then creates
  // Measurements and assigns them to series. On error, crash fatally.
  void AddMeasurementsOrDie(
      SeriesCollection* series_collection,
      absl::flat_hash_map<string, Series*>* type_to_series_map);

  // Find the series that the new measurement should belong to. If the Series
  // does not exist, create it, and add it to the type_to_series_map.
  Series* GetOrCreateSeries(
      SeriesCollection* series_collection,
      absl::flat_hash_map<string, Series*>* type_to_series_map);

  string DebugString() const;

 private:
  void ConsumeCountOrDie(const string& family, const string& name,
                         const string& group = "default");

  void ConsumeBodyMassIndexOrDie(const string& family, const string& name,
                                 const string& group = "default");

  void ConsumePercentageOrDie(const string& family, const string& name,
                              const string& group = "default");

  void ConsumeCountsPerMinuteOrDie(const string& family, const string& name,
                                   const string& group = "default");

  void ConsumeMillilitersPerKilogramMinuteOrDie(
      const string& family, const string& name,
      const string& group = "default");

  void ConsumeKCalOrDie(const string& family, const string& name,
                        const string& group = "default");

  // Distance.

  void ConsumeKilometersOrDie(const string& family, const string& name,
                              const string& group = "default");

  void ConsumeCentimetersOrDie(const string& family, const string& name,
                               const string& group = "default");

  // Volumes.

  void ConsumeMillilitersOrDie(const string& family, const string& name,
                               const string& group = "default");

  // Mass.

  void ConsumeKilogramsOrDie(const string& family, const string& name,
                             const string& group = "default");

  void ConsumeGramsOrDie(const string& family, const string& name,
                         const string& group = "default");

  void ConsumeMilligramsOrDie(const string& family, const string& name,
                              const string& group = "default");

  // Durations.

  void ConsumeMinutesOrDie(const string& family, const string& name,
                           const string& group = "default");

  void ConsumeMillisecondsOrDie(const string& family, const string& name,
                                const string& group = "default");

  void ConsumeDurationOrDie(const string& family, const string& name,
                            const string& group = "default");

  // Categorical measurements: sleep analysis, stand hours, etc.

  void ConsumeSleepAnalysisOrDie(const string& family,
                                 const string& group = "default");

  void ConsumeStandHourOrDie(const string& family,
                             const string& group = "default");

  void ConsumeCountableEventOrDie(const string& family, const string& name,
                                  const string& group = "default");

  Measurement CreateMeasurement(const string& family, const string& name,
                                const string& group, const string& unit,
                                const int64_t value);

  // The string values parsed from the XML Record. Set by InitFromRecordOrDie().
  string type_;
  string unit_;
  string value_;
  string sourceName_;
  string startDate_;
  string endDate_;

  // The series that the XML Record belongs to. Set by
  // AddMeasurementsOrDie().
  Series* series_;

  // Indicates whether the series was created by the call to
  // GetOrCreateSeries(). If true, the Series fields will be set. Else, just
  // the measurement will be added.
  bool new_series_;
};

// Process a SeriesCollection. The source field should be set to the path of
// the XML file to process. Any errors will lead to fatal program crash.
void ProcessHealthKitXmlExportOrDie(SeriesCollection* series_collection);

}  // namespace me
