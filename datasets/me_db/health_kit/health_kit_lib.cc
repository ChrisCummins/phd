#include "datasets/me_db/health_kit/health_kit_lib.h"

#include <boost/property_tree/xml_parser.hpp>

namespace me {

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

void RecordAttributes::ParseFromXmlOrDie(
    const boost::property_tree::ptree& record) {
  int attribute_count = 0;

  for (const boost::property_tree::ptree::value_type& attr :
       record.get_child("<xmlattr>")) {
    if (TryConsumeAttribute(attr, "type", &type_) ||
        TryConsumeAttribute(attr, "unit", &unit_) ||
        TryConsumeAttribute(attr, "value", &value_) ||
        TryConsumeAttribute(attr, "sourceName", &sourceName_) ||
        TryConsumeAttribute(attr, "startDate", &startDate_) ||
        TryConsumeAttribute(attr, "endDate", &endDate_)) {
      ++attribute_count;
    }

    if (attribute_count == 6) {
      return;
    }
  }
  // Not all Records have a unit field. This is the only case in which having
  // less than the full 6 attributes is *not* an error.
  if (!(attribute_count == 5 && unit_.empty()) &&
      !(attribute_count == 4 && unit_.empty() &&
        value_.empty())) {
    FATAL("Failed to parse necessary attributes from Record: %s",
          DebugString());
  }
  return;
}

Series* RecordAttributes::GetOrCreateSeries(
    SeriesCollection* series_collection,
    absl::flat_hash_map<string, Series*>* type_to_series_map) {
  bool* new_series = &new_series_;
  return FindOrAdd<string, Series*>(
        type_to_series_map, type_,
        [&type_to_series_map, series_collection, new_series](
            const string& name) -> Series* {
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

void RecordAttributes::AddMeasurementsFromXmlOrDie(
      const boost::property_tree::ptree& record,
      SeriesCollection* series_collection,
      absl::flat_hash_map<string, Series*>* type_to_series_map) {
  // Set the member variables from the record.
  ParseFromXmlOrDie(record);

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
    FATAL("Unhandled type for record: %s", DebugString());
  }
}

string RecordAttributes::ToString() const {
  return absl::StrFormat("%s %s %s %s %s %s", type_, value_, unit_, sourceName_, startDate_, endDate_);
}

string RecordAttributes::DebugString() const {
  return absl::StrFormat("\ntype=%s\nvalue=%s\nunit=%s\nsource=%s\nstart_date=%s\nend_date=%s", type_, value_, unit_, sourceName_, startDate_, endDate_);
}

void RecordAttributes::ConsumeCountOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "count");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "count", ParseIntOrDie(value_));
}

void RecordAttributes::ConsumeBodyMassIndexOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "count");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "body_mass_index_millis",
      ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumePercentageOrDie(
    const string& family, const string& name, const string& group) {
  if (unit_ != "%") {
    FATAL("Expected unit %%, received unit %s", unit_);
  }
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "percentage_millis",
      ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumeCountsPerMinuteOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "count/min");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "beats_per_minute_millis",
      ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumeMillilitersPerKilogramMinuteOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "mL/min·kg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group,
      "milliliters_per_kilogram_per_minute_millis",
      ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumeKCalOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "kcal");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "calories", ParseDoubleOrDie(value_) * 1000);
}

void RecordAttributes::ConsumeKilometersOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "km");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "millimeters", ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumeCentimetersOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "cm");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "millimeters", ParseDoubleOrDie(value_) * 10);
}

void RecordAttributes::ConsumeMillilitersOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "mL");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliliters", ParseIntOrDie(value_));
}

void RecordAttributes::ConsumeKilogramsOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "kg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_) * 1000000);
}

void RecordAttributes::ConsumeGramsOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "g");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_) * 1000);
}

void RecordAttributes::ConsumeMilligramsOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "mg");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milligrams", ParseDoubleOrDie(value_));
}

void RecordAttributes::ConsumeMinutesOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "min");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliseconds",
      ParseDoubleOrDie(value_) * 60 * 1000);
}

void RecordAttributes::ConsumeMillisecondsOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_ == "ms");
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliseconds", ParseDoubleOrDie(value_));
}

void RecordAttributes::ConsumeDurationOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(value_.empty());
  CHECK(unit_.empty());
  int64_t duration_ms = ParseDateOrDie(endDate_) - ParseDateOrDie(startDate_);
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliseconds", duration_ms);
}

void RecordAttributes::ConsumeSleepAnalysisOrDie(
    const string& family, const string& group) {
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
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "milliseconds", duration_ms);
}

void RecordAttributes::ConsumeStandHourOrDie(
    const string& family, const string& group) {
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
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "count", 1);
}

void RecordAttributes::ConsumeCountableEventOrDie(
    const string& family, const string& name, const string& group) {
  CHECK(unit_.empty());
  *series_->add_measurement() = CreateMeasurement(
      family, name, group, "count", 1);
}

Measurement RecordAttributes::CreateMeasurement(
    const string& family, const string& name, const string& group,
    const string& unit, const int64_t value) {
  if (new_series_) {
    series_->set_name(name);
    series_->set_family(family);
    series_->set_unit(unit);
    new_series_ = false;
  }

  Measurement measurement;
  measurement.set_ms_since_unix_epoch(ParseDateOrDie(startDate_));
  measurement.set_value(value);
  measurement.set_group(group);

  // Set the source as the device name.
  CHECK(!sourceName_.empty());
  measurement.set_source(
      absl::StrFormat("HealthKit:%s", phd::ToCamelCase(sourceName_)));

  return measurement;
}


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
      RecordAttributes record;
      record.AddMeasurementsFromXmlOrDie(health_elem.second, series_collection,
                                         &type_to_series_map);
    }
  }
}

}  // namespace me
