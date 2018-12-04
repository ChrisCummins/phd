// Protocol buffer processing binary for extracting a me.Series proto from a
// YNAB JSON file.
#include "phd/macros.h"
#include "phd/pbutil.h"
#include "phd/string.h"

#include "util/me/me.pb.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/container/flat_hash_map.h"

#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

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

int64_t ParseDateOrDie(const string& date) {
  absl::Time time;
  std::string err;
  bool succeeded = absl::ParseTime("%Y-%m-%d %H:%M:%S %z", date, &time, &err);
  if (!succeeded) {
    FATAL("Failed to parse date '%s': %s", date, err);
  }
  absl::Duration d = time - absl::UnixEpoch();
  return d / absl::Milliseconds(1);
}

// If the given attribute's name matches attribute_name, set attribute_value.
// Returns true if the attribute_value was set, else false. The attribute_value
// must be a pointer to an empty string.
bool TryConsumeAttribute(
    const boost::property_tree::ptree::value_type& attribute,
    const string attribute_name, string* attribute_value) {
  if (attribute.first != attribute_name) {
    return false;
  }
  CHECK(attribute_value->empty());
  *attribute_value = attribute.second.data();
  return true;
}

int64_t ParseIntOrDie(const string& integer_string) {
  char* endptr;
  int64_t number = std::strtol(integer_string.c_str(), &endptr, 10);
  if (endptr == integer_string.c_str() || *endptr != '\0') {
    // Not a valid number at all
    FATAL("Cannot convert string to integer: `%s`", integer_string);
  }
  return number;
}

double ParseDoubleOrDie(const string& double_string) {
  char* endptr;
  double number = std::strtod(double_string.c_str(), &endptr);
  if (endptr == double_string.c_str() || *endptr != '\0') {
    FATAL("Cannot convert string to double: `%s`", double_string);
  }
  return number;
}

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
  string type_;
  string unit_;
  string value_;
  string sourceName_;
  string startDate_;
  string endDate_;

  string ToString() const {
    return absl::StrFormat("%s %s %s %s %s %s", type_, value_, unit_, sourceName_, startDate_, endDate_);
  }

  string DebugString() const {
    return absl::StrFormat("\ntype=%s\nvalue=%s\nunit=%s\nsource=%s\nstart_date=%s\nend_date=%s", type_, value_, unit_, sourceName_, startDate_, endDate_);
  }

  void AddMeasurementToSeries(Series*const series, const bool new_series) {
    // Create the new measurement.
    series_ = series;
    measurement_ = series->add_measurement();
    new_series_ = new_series;

    if (type_ == "HKQuantityTypeIdentifierDietaryWater") {
      ConsumeMillilitersOrDie("Diet", "WaterConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierBodyMassIndex") {
      ConsumeBodyMassIndexOrDie("BodyMeasurements", "BodyMassIndex");
    } else if (type_ == "HKQuantityTypeIdentifierHeight") {
      ConsumeCentimetersOrDie("BodyMeasurements", "Height");
    } else if (type_ == "HKQuantityTypeIdentifierBodyMass") {
      ConsumeKilogramsOrDie("BodyMeasurements", "Weight");
    } else if (type_ == "HKQuantityTypeIdentifierHeartRate") {
      ConsumeCountsPerMinuteOrDie("BodyMeasurements", "HeartRate");
    } else if (type_ == "HKQuantityTypeIdentifierBodyFatPercentage") {
      ConsumePercentageOrDie("BodyMeasurements", "BodyFatPercentage");
    } else if (type_ == "HKQuantityTypeIdentifierLeanBodyMass") {
      ConsumeKilogramsOrDie("BodyMeasurements", "LeanBodyMass");
    } else if (type_ == "HKQuantityTypeIdentifierStepCount") {
      ConsumeCountOrDie("Activity", "StepCount");
    } else if (type_ == "HKQuantityTypeIdentifierDistanceWalkingRunning") {
      ConsumeKilometersOrDie("Activity", "WalkingRunningDistance");
    } else if (type_ == "HKQuantityTypeIdentifierBasalEnergyBurned") {
      ConsumeKCalOrDie("Activity", "RestingEnergy");
    } else if (type_ == "HKQuantityTypeIdentifierActiveEnergyBurned") {
      ConsumeKCalOrDie("Activity", "ActiveEnergy");
    } else if (type_ == "HKQuantityTypeIdentifierFlightsClimbed") {
      ConsumeCountOrDie("Activity", "FlightClimbed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryFatTotal") {
      ConsumeGramsOrDie("Diet", "TotalFatConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryFatSaturated") {
      ConsumeGramsOrDie("Diet", "SaturatedFatConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryCholesterol") {
      ConsumeMilligramsOrDie("Diet", "CholesterolConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietarySodium") {
      ConsumeMilligramsOrDie("Diet", "SodiumConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryCarbohydrates") {
      ConsumeGramsOrDie("Diet", "CarbohydratesConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryFiber") {
      ConsumeGramsOrDie("Diet", "FiberConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierAppleExerciseTime") {
      ConsumeMinutesOrDie("TimeTracking", "ExerciseTime");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryCaffeine") {
      ConsumeMilligramsOrDie("Diet", "CaffeineConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDistanceCycling") {
      ConsumeKilometersOrDie("Activity", "DistanceCycling");
    } else if (type_ == "HKQuantityTypeIdentifierRestingHeartRate") {
      ConsumeCountsPerMinuteOrDie("BodyMeasurements", "RestingHeartRate");
    } else if (type_ == "HKQuantityTypeIdentifierVO2Max") {
      ConsumeMillilitersPerKilogramMinuteOrDie("BodyMeasurements", "VO2Max");
    } else if (type_ == "HKQuantityTypeIdentifierWalkingHeartRateAverage") {
      ConsumeCountsPerMinuteOrDie("BodyMeasurements", "WalkingHeartRateAvg");
    } else if (type_ == "HKCategoryTypeIdentifierSleepAnalysis") {
      ConsumeSleepAnalysisOrDie("Activity");
    } else if (type_ == "HKCategoryTypeIdentifierAppleStandHour") {
      ConsumeStandHourOrDie("Activity");
    } else if (type_ == "HKCategoryTypeIdentifierSexualActivity") {
      ConsumeCountableEventOrDie("Activity", "SexualActivityCount");
    } else if (type_ == "HKCategoryTypeIdentifierMindfulSession") {
      ConsumeDurationOrDie("TimeTracking", "MindfulnessTime");
    } else if (type_ == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN") {
      ConsumeMillisecondsOrDie("BodyMeasurements", "HeartRateVariability");
    } else if (type_ == "HKQuantityTypeIdentifierDietarySugar") {
      ConsumeGramsOrDie("Diet", "SugarConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryEnergyConsumed") {
      ConsumeKCalOrDie("Diet", "CaloriesConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryProtein") {
      ConsumeGramsOrDie("Diet", "ProteinConsumed");
    } else if (type_ == "HKQuantityTypeIdentifierDietaryPotassium") {
      ConsumeMilligramsOrDie("Diet", "PotassiumConsumed");
//    } else if (type_ == "TODO") {
//      ConsumeMillisecondsOrDie("FAMILY", "NAME");
//    } else if (type_ == "TODO") {
//      ConsumeMillisecondsOrDie("FAMILY", "NAME");
//    } else if (type_ == "TODO") {
//      ConsumeMillisecondsOrDie("FAMILY", "NAME");
    } else {
      FATAL("Unhandled type for record: %s", DebugString());
    }
  }

  static RecordAttributes CreateFromXmlRecord(
      const boost::property_tree::basic_ptree<std::__1::basic_string<char>, std::__1::basic_string<char>, std::__1::less<std::__1::basic_string<char> > >& record) {
    RecordAttributes attributes;
    int attribute_count = 0;

    for (const boost::property_tree::ptree::value_type& attr :
         record.get_child("<xmlattr>")) {
      if (TryConsumeAttribute(attr, "type", &attributes.type_) ||
          TryConsumeAttribute(attr, "unit", &attributes.unit_) ||
          TryConsumeAttribute(attr, "value", &attributes.value_) ||
          TryConsumeAttribute(attr, "sourceName", &attributes.sourceName_) ||
          TryConsumeAttribute(attr, "startDate", &attributes.startDate_) ||
          TryConsumeAttribute(attr, "endDate", &attributes.endDate_)) {
        ++attribute_count;
      }

      if (attribute_count == 6) {
        return attributes;
      }
    }
    // Not all Records have a unit field. This is the only case in which having
    // less than the full 6 attributes is *not* an error.
    if (!(attribute_count == 5 && attributes.unit_.empty()) &&
        !(attribute_count == 4 && attributes.unit_.empty() &&
          attributes.value_.empty())) {
      FATAL("Failed to parse necessary attributes from Record: %s",
            attributes.DebugString());
    }
    return attributes;
  }

 private:
  void ConsumeCountOrDie(const string& family, const string& name,
                         const string& group = "default") {
    CHECK(unit_ == "count");
    SetMeasurement(family, name, group, /*unit=*/"count",
                   /*value=*/ParseIntOrDie(value_));
  }

 void ConsumeBodyMassIndexOrDie(const string& family, const string& name,
                                const string& group = "default") {
    CHECK(unit_ == "count");
    SetMeasurement(family, name, group, /*unit=*/"body_mass_index_millis",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumePercentageOrDie(const string& family, const string& name,
                              const string& group = "default") {
    if (unit_ != "%") {
      FATAL("Expected unit %%, received unit %s", unit_);
    }
    SetMeasurement(family, name, group, /*unit=*/"percentage_millis",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumeCountsPerMinuteOrDie(const string& family, const string& name,
                                  const string& group = "default") {
    CHECK(unit_ == "count/min");
    SetMeasurement(family, name, group, /*unit=*/"beats_per_minute_millis",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumeMillilitersPerKilogramMinuteOrDie(
      const string& family, const string& name,
      const string& group = "default") {
    CHECK(unit_ == "mL/min·kg");
    SetMeasurement(family, name, group,
                   /*unit=*/"milliliters_per_kilogram_per_minute_millis",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumeKCalOrDie(const string& family, const string& name,
                        const string& group = "default") {
    CHECK(unit_ == "kcal");
    SetMeasurement(family, name, group, /*unit=*/"calories",
                   /*value=*/ParseDoubleOrDie(value_) * 1000);
  }

  // Distance.

  void ConsumeKilometersOrDie(const string& family, const string& name,
                              const string& group = "default") {
    CHECK(unit_ == "km");
    SetMeasurement(family, name, group, /*unit=*/"millimeters",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumeCentimetersOrDie(const string& family, const string& name,
                               const string& group = "default") {
    CHECK(unit_ == "cm");
    SetMeasurement(family, name, group, /*unit=*/"millimeters",
                   /*value=*/ParseDoubleOrDie(value_) * 10);
  }

  // Volumes.

  void ConsumeMillilitersOrDie(const string& family, const string& name,
                               const string& group = "default") {
    CHECK(unit_ == "mL");
    SetMeasurement(family, name, group, /*unit=*/"milliliters",
                   /*value=*/ParseIntOrDie(value_));
  }

  // Mass.

  void ConsumeKilogramsOrDie(const string& family, const string& name,
                             const string& group = "default") {
    CHECK(unit_ == "kg");
    SetMeasurement(family, name, group, /*unit=*/"milligrams",
                   /*value=*/ParseDoubleOrDie(value_) * 1000000);
  }

  void ConsumeGramsOrDie(const string& family, const string& name,
                           const string& group = "default") {
    CHECK(unit_ == "g");
    SetMeasurement(family, name, group, /*unit=*/"milligrams",
                   /*value=*/ParseDoubleOrDie(value_) * 1000);
  }

  void ConsumeMilligramsOrDie(const string& family, const string& name,
                              const string& group = "default") {
    CHECK(unit_ == "mg");
    SetMeasurement(family, name, group, /*unit=*/"milligrams",
                   /*value=*/ParseDoubleOrDie(value_));
  }

  // Durations.

  void ConsumeMinutesOrDie(const string& family, const string& name,
                           const string& group = "default") {
    CHECK(unit_ == "min");
    SetMeasurement(family, name, group, /*unit=*/"milliseconds",
                   /*value=*/ParseDoubleOrDie(value_) * 60 * 1000);
  }

  void ConsumeMillisecondsOrDie(const string& family, const string& name,
                                const string& group = "default") {
      CHECK(unit_ == "ms");
      SetMeasurement(family, name, group, /*unit=*/"milliseconds",
                     /*value=*/ParseDoubleOrDie(value_));
  }

  void ConsumeDurationOrDie(const string& family, const string& name,
                            const string& group = "default") {
    CHECK(value_.empty());
    CHECK(unit_.empty());
    int64_t duration_ms = ParseDateOrDie(endDate_) - ParseDateOrDie(startDate_);
    SetMeasurement(family, name, group, /*unit=*/"milliseconds",
                   /*value=*/duration_ms);
  }

  // Categorical measurements: sleep analysis, stand hours, etc.

  void ConsumeSleepAnalysisOrDie(const string& family,
                                 const string& group = "default") {
    CHECK(unit_.empty());
    string name;
    if (value_ == "HKCategoryValueSleepAnalysisAsleep") {
      name = "SleepTime";
    } else if (value_ == "HKCategoryValueSleepAnalysisInBed") {
      name = "InBedTime";
    } else if (value_ == "HKCategoryValueSleepAnalysisAwake") {
      name = "AwakeTime";
    } else {
      FATAL("Could not handle the value field of "
            "sleep analysis Record: %s", DebugString());
    }
    int64_t duration_ms = ParseDateOrDie(endDate_) - ParseDateOrDie(startDate_);
    SetMeasurement(family, name, group, /*unit=*/"milliseconds",
                   /*value=*/duration_ms);
  }

  void ConsumeStandHourOrDie(const string& family,
                             const string& group = "default") {
    CHECK(unit_.empty());
    string name;
    if (value_ == "HKCategoryValueAppleStandHourIdle") {
      name = "IdleHours";
    } else if (value_ == "HKCategoryValueAppleStandHourStood") {
      name = "StandHours";
    } else {
      FATAL("Could not handle the value field of "
            "stand hour Record: %s", DebugString());
    }
    SetMeasurement(family, name, group, /*unit=*/"count", /*value=*/1);
  }

  void ConsumeCountableEventOrDie(const string& family, const string& name,
                                  const string& group = "default") {
    CHECK(unit_.empty());
    SetMeasurement(family, name, group, /*unit=*/"count", /*value=*/1);
  }

  void SetMeasurement(const string& family, const string& name,
                      const string& group, const string& unit,
                      const int64_t value) {
    if (new_series_) {
      series_->set_name(name);
      series_->set_family(family);
      series_->set_unit(unit);
    }
    measurement_->set_ms_since_unix_epoch(ParseDateOrDie(startDate_));
    measurement_->set_value(value);
    measurement_->set_group(group);

    // Set the source as the device name.
    CHECK(!sourceName_.empty());
    measurement_->set_source(
        absl::StrFormat("HealthKit:%s", phd::ToCamelCase(sourceName_)));
  }

  Series* series_;
  Measurement* measurement_;
  bool new_series_;
};


void ProcessHealthKitXmlExport(SeriesCollection* series_collection) {
  const boost::filesystem::path xml_path(series_collection->source());

  CHECK(boost::filesystem::is_regular_file(xml_path));
  INFO("Reading from XML file %s", xml_path.string());

  boost::filesystem::ifstream xml(xml_path);
  CHECK(xml.is_open());

  boost::property_tree::ptree root;
  boost::property_tree::read_xml(xml, root);

  // Keep a map from Record.type to series. Measurements are assigned to named
  // Series. We use this map to determine which Series to add each Measurement
  // to.
  absl::flat_hash_map<string, Series*> type_to_series_map;

  // Iterate over all "HealthData" elements.
  int record_index = 0;
  for (const boost::property_tree::ptree::value_type& health_elem :
       root.get_child("HealthData")) {

    // There are multiple types for HealthData elements. We're only interested
    // in records.
    if (health_elem.first != "Record") {
      continue;
    }

    ++record_index;
    RecordAttributes record = RecordAttributes::CreateFromXmlRecord(
        health_elem.second);

    bool set_properties_on_series = false;
    // Add a pointer to the flag so that the FindOrAdd lambda can capture it by
    // reference.
    bool *new_series = &set_properties_on_series;

    // Find the series that the new measurement should belong to. If the Series
    // does not exist, create it.
    Series* series = FindOrAdd<string, Series*>(
        &type_to_series_map, record.type_,
        [&type_to_series_map,series_collection,new_series](
            const string& name) -> Series* {
      *new_series = true;
      Series* series = series_collection->add_series();
      type_to_series_map.insert(
          std::make_pair(name, series));
      return series;
    });

    record.AddMeasurementToSeries(series, set_properties_on_series);
  }
}

}  // namespace me

PBUTIL_INPLACE_PROCESS_MAIN(me::ProcessHealthKitXmlExport, me::SeriesCollection);
