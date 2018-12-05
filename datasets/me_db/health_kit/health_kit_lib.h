// Library for processing HealthKit export.xml files.
//
// See //datasets/me_db/health_kit:README.md for an overview of the schema that
// is processed, and the measurements produced.
//
#pragma once

#include "datasets/me_db/me.pb.h"

#include "phd/macros.h"
#include "phd/string.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/container/flat_hash_map.h"

#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>

namespace me {

template<typename K, typename V>
V FindOrAdd(absl::flat_hash_map<K, V>* map, const K& key,
            std::function<V(const K&)> add_callback) {
  auto it = map->find(key);

  if (it == map->end()) {
    return add_callback(key);
  } else {
    return it->second;
  }
}

int64_t ParseDateOrDie(const string& date);

// If the given attribute's name matches attribute_name, set attribute_value.
// Returns true if the attribute_value was set, else false. The attribute_value
// must be a pointer to an empty string.
bool TryConsumeAttribute(
    const boost::property_tree::ptree::value_type& attribute,
    const string attribute_name, string* attribute_value);

int64_t ParseIntOrDie(const string& integer_string);

double ParseDoubleOrDie(const string& double_string);

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
class RecordAttributes {
 public:
  RecordAttributes() : new_series_(false) {}

  // Initialize values from an XML Record entry. On error, crash fatally.
  void AddMeasurementsFromXmlOrDie(
      const boost::property_tree::ptree& record,
      SeriesCollection* series_collection,
      absl::flat_hash_map<string, Series*>* type_to_series_map);

  void ParseFromXmlOrDie(const boost::property_tree::ptree& record);

  // Find the series that the new measurement should belong to. If the Series
  // does not exist, create it, and add it to the type_to_series_map.
  Series* GetOrCreateSeries(
      SeriesCollection* series_collection,
      absl::flat_hash_map<string, Series*>* type_to_series_map);

  string ToString() const;

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

  Measurement CreateMeasurement(
      const string& family, const string& name, const string& group,
      const string& unit, const int64_t value);

  // The string values parsed from the XML Record. Set by ParseFromXmlOrDie().
  string type_;
  string unit_;
  string value_;
  string sourceName_;
  string startDate_;
  string endDate_;

  // The series that the XML Record belongs to. Set by
  // AddMeasurementsFromXmlOrDie().
  Series* series_;

  // Indicates whether the series was created by the call to
  // GetOrCreateSeries(). If true, the Series fields will be set. Else, just
  // the measurement will be added.
  bool new_series_;
};

void ProcessHealthKitXmlExport(SeriesCollection* series_collection);

}  // namespace me
