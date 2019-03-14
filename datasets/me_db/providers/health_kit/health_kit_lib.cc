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
#include "datasets/me_db/providers/health_kit/health_kit_lib.h"

#include "phd/logging.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"

#include <boost/property_tree/xml_parser.hpp>

namespace me {

int64_t ParseHealthKitDatetimeOrDie(const string& date) {
  absl::Time time;
  std::string err;
  bool succeeded = absl::ParseTime("%Y-%m-%d %H:%M:%S %z", date, &time, &err);
  if (!succeeded) {
    LOG(FATAL) << "Failed to parse HealthKit datetime '" << date
               << "': " << err;
  }
  absl::Duration d = time - absl::UnixEpoch();
  return d / absl::Milliseconds(1);
}

bool SetAttributeIfMatch(
    const boost::property_tree::ptree::value_type& attribute,
    const string& attribute_name, string* attribute_value) {
  if (attribute.first == attribute_name) {
    *attribute_value = attribute.second.data();
    return true;
  }
  return false;
}

int64_t ParseIntOrDie(const string& integer_string) {
  char* endptr;
  int64_t number = std::strtol(integer_string.c_str(), &endptr, 10);
  if (endptr == integer_string.c_str() || *endptr != '\0') {
    // Not a valid number at all
    LOG(FATAL) << "Failed to parse integer '" << integer_string << "'";
  }
  return number;
}

double ParseDoubleOrDie(const string& double_string) {
  char* endptr;
  double number = std::strtod(double_string.c_str(), &endptr);
  if (endptr == double_string.c_str() || *endptr != '\0') {
    LOG(FATAL) << "Failed to parse double '" << double_string << "'";
  }
  return number;
}

void HealthKitRecordImporter::InitFromRecordOrDie(
    const boost::property_tree::ptree& record) {
  int attribute_count = 0;

  // Clear the member variables which might not be set by the for loop below.
  // This allows instances of this class to be reused to process multiple
  // records. Member variables not cleared here *must* cause an error if they
  // are not found in the attributes of the record currently being processed.
  unit_.clear();
  value_.clear();

  for (const boost::property_tree::ptree::value_type& attr :
       record.get_child("<xmlattr>")) {
    if (SetAttributeIfMatch(attr, "type", &type_) ||
        SetAttributeIfMatch(attr, "unit", &unit_) ||
        SetAttributeIfMatch(attr, "value", &value_) ||
        SetAttributeIfMatch(attr, "sourceName", &sourceName_) ||
        SetAttributeIfMatch(attr, "startDate", &startDate_) ||
        SetAttributeIfMatch(attr, "endDate", &endDate_)) {
      ++attribute_count;
    }

    // If we have already found the six fields we are interested in, we can
    // break out of the loop early.
    if (attribute_count == 6) {
      return;
    }
  }

  // Not all Records have a unit or value field, so it is possible to iterate
  // over all attributes without finding these values.
  if (!(attribute_count == 5 && unit_.empty()) &&
      !(attribute_count == 4 && unit_.empty() && value_.empty())) {
    LOG(FATAL) << "Failed to parse necessary attributes from Record: "
               << DebugString();
  }
}

Series* HealthKitRecordImporter::GetOrCreateSeries(
    SeriesCollection* series_collection,
    absl::flat_hash_map<string, Series*>* type_to_series_map) {
  bool* new_series = &new_series_;
  return FindOrAdd<string, Series*>(
      type_to_series_map, type_,
      [&type_to_series_map, series_collection,
       new_series](const string& name) -> Series* {
        // Create the new series. We don't set series field values immediately,
        // since we handle the conversion from XML Record values to our values
        // at the same time we create measurements. So instead we set the
        // new_series_ member variable to true and defer until SetMeasurement()
        // is called.
        Series* series = series_collection->add_series();
        *new_series = true;
        type_to_series_map->insert(std::make_pair(name, series));
        return series;
      });
}

void HealthKitRecordImporter::AddMeasurementsOrDie(
    SeriesCollection* series_collection,
    absl::flat_hash_map<string, Series*>* type_to_series_map) {
  // Get the series to associate with new measurements.
  series_ = GetOrCreateSeries(series_collection, type_to_series_map);

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
  } else {
    LOG(FATAL) << "Unhandled type '" << type_
               << "' for HealthKit record: " << DebugString();
  }
}

string HealthKitRecordImporter::DebugString() const {
  return absl::StrFormat(
      "type=%s\nvalue=%s\nunit=%s\nsource=%s\nstart_date=%s\nend_date=%s",
      type_, value_, unit_, sourceName_, startDate_, endDate_);
}

void HealthKitRecordImporter::ConsumeCountOrDie(const string& family,
                                                const string& name,
                                                const string& group) {
  CHECK(unit_ == "count");
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "count", ParseIntOrDie(value_));
}

void HealthKitRecordImporter::ConsumeBodyMassIndexOrDie(const string& family,
                                                        const string& name,
                                                        const string& group) {
  CHECK(unit_ == "count");
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "body_mass_index_millis",
                        ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumePercentageOrDie(const string& family,
                                                     const string& name,
                                                     const string& group) {
  if (unit_ != "%") {
    LOG(FATAL) << "Expected unit %, received unit " << unit_;
  }
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "percentage_millis",
                        ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumeCountsPerMinuteOrDie(const string& family,
                                                          const string& name,
                                                          const string& group) {
  CHECK(unit_ == "count/min");
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "beats_per_minute_millis",
                        ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumeMillilitersPerKilogramMinuteOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "mL/min·kg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliliters_per_kilogram_per_minute_millis",
      ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumeKCalOrDie(const string& family,
                                               const string& name,
                                               const string& group) {
  CHECK(unit_ == "kcal");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "calories", ParseDoubleOrDie(value_) * 1000);
}

void HealthKitRecordImporter::ConsumeKilometersOrDie(const string& family,
                                                     const string& name,
                                                     const string& group) {
  CHECK(unit_ == "km");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "millimeters", ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumeCentimetersOrDie(const string& family,
                                                      const string& name,
                                                      const string& group) {
  CHECK(unit_ == "cm");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "millimeters", ParseDoubleOrDie(value_) * 10);
}

void HealthKitRecordImporter::ConsumeMillilitersOrDie(const string& family,
                                                      const string& name,
                                                      const string& group) {
  CHECK(unit_ == "mL");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliliters", ParseIntOrDie(value_));
}

void HealthKitRecordImporter::ConsumeKilogramsOrDie(const string& family,
                                                    const string& name,
                                                    const string& group) {
  CHECK(unit_ == "kg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_) * 1000000);
}

void HealthKitRecordImporter::ConsumeGramsOrDie(const string& family,
                                                const string& name,
                                                const string& group) {
  CHECK(unit_ == "g");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_) * 1000);
}

void HealthKitRecordImporter::ConsumeMilligramsOrDie(const string& family,
                                                     const string& name,
                                                     const string& group) {
  CHECK(unit_ == "mg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_));
}

void HealthKitRecordImporter::ConsumeMinutesOrDie(const string& family,
                                                  const string& name,
                                                  const string& group) {
  CHECK(unit_ == "min");
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "milliseconds",
                        ParseDoubleOrDie(value_) * 60 * 1000);
}

void HealthKitRecordImporter::ConsumeMillisecondsOrDie(const string& family,
                                                       const string& name,
                                                       const string& group) {
  CHECK(unit_ == "ms");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliseconds", ParseDoubleOrDie(value_));
}

void HealthKitRecordImporter::ConsumeDurationOrDie(const string& family,
                                                   const string& name,
                                                   const string& group) {
  CHECK(value_.empty());
  CHECK(unit_.empty());
  int64_t duration_ms = ParseHealthKitDatetimeOrDie(endDate_) -
                        ParseHealthKitDatetimeOrDie(startDate_);
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "milliseconds", duration_ms);
}

void HealthKitRecordImporter::ConsumeSleepAnalysisOrDie(const string& family,
                                                        const string& group) {
  CHECK(unit_.empty());
  string name;
  if (value_ == "HKCategoryValueSleepAnalysisAsleep") {
    name = "SleepTime";
  } else if (value_ == "HKCategoryValueSleepAnalysisInBed") {
    name = "InBedTime";
  } else if (value_ == "HKCategoryValueSleepAnalysisAwake") {
    name = "AwakeTime";
  } else {
    LOG(FATAL) << "Could not handle the value field of sleep analysis Record: "
               << DebugString();
  }
  int64_t duration_ms = ParseHealthKitDatetimeOrDie(endDate_) -
                        ParseHealthKitDatetimeOrDie(startDate_);
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "milliseconds", duration_ms);
}

void HealthKitRecordImporter::ConsumeStandHourOrDie(const string& family,
                                                    const string& group) {
  CHECK(unit_.empty());
  string name;
  if (value_ == "HKCategoryValueAppleStandHourIdle") {
    name = "IdleHours";
  } else if (value_ == "HKCategoryValueAppleStandHourStood") {
    name = "StandHours";
  } else {
    LOG(FATAL) << "Could not handle the value field of stand hour Record: "
               << DebugString();
  }
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "count", 1);
}

void HealthKitRecordImporter::ConsumeCountableEventOrDie(const string& family,
                                                         const string& name,
                                                         const string& group) {
  CHECK(unit_.empty());
  *series_->add_measurement() =
      CreateMeasurement(family, name, group, "count", 1);
}

Measurement HealthKitRecordImporter::CreateMeasurement(const string& family,
                                                       const string& name,
                                                       const string& group,
                                                       const string& unit,
                                                       const int64_t value) {
  if (new_series_) {
    series_->set_name(name);
    series_->set_family(family);
    series_->set_unit(unit);
    new_series_ = false;
  }

  Measurement measurement;
  measurement.set_ms_since_unix_epoch(ParseHealthKitDatetimeOrDie(startDate_));
  measurement.set_value(value);
  measurement.set_group(group);

  // Set the source as the device name.
  CHECK(!sourceName_.empty());
  measurement.set_source(
      absl::StrFormat("HealthKit:%s", phd::ToCamelCase(sourceName_)));

  return measurement;
}

void ProcessHealthKitXmlExportOrDie(SeriesCollection* series_collection) {
  const boost::filesystem::path xml_path(series_collection->source());

  CHECK(boost::filesystem::is_regular_file(xml_path));
  LOG(INFO) << "Reading from XML file " << xml_path.string();

  boost::filesystem::ifstream xml(xml_path);
  CHECK(xml.is_open());

  boost::property_tree::ptree root;
  boost::property_tree::read_xml(xml, root);

  // Keep a map from Record.type to series. Measurements are assigned to named
  // Series. We use this map to determine which Series to add each Measurement
  // to.
  absl::flat_hash_map<string, Series*> type_to_series_map;

  HealthKitRecordImporter record_importer;

  // Iterate over all "HealthData" elements.
  for (const boost::property_tree::ptree::value_type& health_elem :
       root.get_child("HealthData")) {
    // There are multiple types for HealthData elements. We're only interested
    // in records.
    // TODO(cec): Add support for ActivitySummary and Workout elements.
    //
    // Schema for workouts:
    //
    //     <!ATTLIST Workout
    //       workoutActivityType   CDATA #REQUIRED
    //       duration              CDATA #IMPLIED
    //       durationUnit          CDATA #IMPLIED
    //       totalDistance         CDATA #IMPLIED
    //       totalDistanceUnit     CDATA #IMPLIED
    //       totalEnergyBurned     CDATA #IMPLIED
    //       totalEnergyBurnedUnit CDATA #IMPLIED
    //       sourceName            CDATA #REQUIRED
    //       sourceVersion         CDATA #IMPLIED
    //       device                CDATA #IMPLIED
    //       creationDate          CDATA #IMPLIED
    //       startDate             CDATA #REQUIRED
    //       endDate               CDATA #REQUIRED
    //     >
    //
    // Schema for activity summaries:
    //
    //     <!ATTLIST ActivitySummary
    //       dateComponents           CDATA #IMPLIED
    //       activeEnergyBurned       CDATA #IMPLIED
    //       activeEnergyBurnedGoal   CDATA #IMPLIED
    //       activeEnergyBurnedUnit   CDATA #IMPLIED
    //       appleExerciseTime        CDATA #IMPLIED
    //       appleExerciseTimeGoal    CDATA #IMPLIED
    //       appleStandHours          CDATA #IMPLIED
    //       appleStandHoursGoal      CDATA #IMPLIED
    //     >
    if (health_elem.first == "Record") {
      record_importer.InitFromRecordOrDie(health_elem.second);
      record_importer.AddMeasurementsOrDie(series_collection,
                                           &type_to_series_map);
    }
  }
}

}  // namespace me
