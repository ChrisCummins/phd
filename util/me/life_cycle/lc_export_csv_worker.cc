// Process a Life Cycle CSV file and extract series.
#include "phd/macros.h"
#include "phd/pbutil.h"
#include "phd/string.h"

#include "util/me/me.pb.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/container/flat_hash_map.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

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


int64_t ParseTimeOrDie(const string& date) {
  absl::Time time;
  std::string err;
  bool succeeded = absl::ParseTime("%Y-%m-%d %H:%M:%S", date, &time, &err);
  if (!succeeded) {
    FATAL("Failed to parse '%s': %s", date, err);
  }
  absl::Duration d = time - absl::UnixEpoch();
  return d / absl::Milliseconds(1);
}


void ProcessLcExportCsv(SeriesCollection* proto) {
  const boost::filesystem::path csv_path(proto->source());

  CHECK(boost::filesystem::is_regular_file(csv_path));
  INFO("Reading from CSV file %s", csv_path.string());

  boost::filesystem::ifstream csv(csv_path);
  CHECK(csv.is_open());

  string line;
  // Skip the first line which is the header.
  std::getline(csv, line);
  // Skip the second line, which should be empty.
  std::getline(csv, line);
  CHECK(phd::TrimRightCopy(line).empty());

  // Keep a map from name columns to series. Measurements are assigned to named
  // Series. We use this map to determine which Series to add each Measurement
  // to.
  absl::flat_hash_map<string, Series*> name_to_series_map;

  // Iterate through the file.
  int line_num = 2;  // We skipped the first two lines.
  while (std::getline(csv, line)) {
    ++line_num;
    // Split the comma separated line.
    std::vector<absl::string_view> components = absl::StrSplit(line, ',');
    if (components.size() < 8) {
      FATAL("Line %d of %s does not have 8 columns: '%s'",
            line_num, csv_path.string(), line);
    }

    // Split out the variables from the line.
    const string start_date_str = string(components[0]);
    const string end_date_str = string(components[1]);
    const string name = phd::TrimCopy(string(components[5]));
    const string location = phd::TrimCopy(string(components[6]));

    // Parse the timestamps.
    int64_t start_time = ParseTimeOrDie(start_date_str);
    int64_t end_time = ParseTimeOrDie(end_date_str);
    CHECK(start_time);
    CHECK(end_time);

    // Find the series that the new measurement should belong to. If the Series
    // does not exist, create it.
    Series* series = FindOrAdd<string, Series*>(
        &name_to_series_map, name,
        [&](const string& name) -> Series* {
      Series* series = proto->add_series();
      series->set_name(absl::StrCat(phd::ToCamelCase(name), "Time"));
      series->set_family("TimeTracking");
      series->set_unit("milliseconds");
      name_to_series_map.insert(
          std::make_pair(name, series));
      return series;
    });

    // Create the new measurement.
    Measurement* measurement = series->add_measurement();
    measurement->set_ms_since_epoch_utc(start_time);
    measurement->set_value(end_time - start_time);
    if (location.empty()) {
      measurement->set_group("default");
    } else {
      measurement->set_group(phd::ToCamelCase(location));
    }
    measurement->set_source("LifeCycle");
  }
}

}  // namespace me

PBUTIL_INPLACE_PROCESS_MAIN(me::ProcessLcExportCsv, me::SeriesCollection);
