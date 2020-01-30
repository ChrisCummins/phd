// Library for processing Life Cycle CSV files.
//
// See //datasets/me_db/providers/life_cycle:README.md for an overview of the
// schema that is processed, and the measurements produced.
//
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
#pragma once

#include "labm8/cpp/macros.h"
#include "labm8/cpp/string.h"

#include "datasets/me_db/me.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

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

// Round a timestamp, as milliseconds since the Unix epoch, up to the "zeroth"
// millisecond of the next day.
int64_t RoundToStartOfNextDay(const int64_t ms_since_unix_epoch);

// Parse a (START|END)_DATE or (START|END)_TIME column to an absl::Time
// instance, or fatally error.
absl::Time ParseLifeCycleDatetimeOrDie(const string& date);

// Convert an absl::Time instance to the number of milliseconds since the Unix
// epoch.
int64_t ToMillisecondsSinceUnixEpoch(const absl::Time& time);

// Create measurements for the duration. If the duration overflows to
// subsequent dates, it is split into multiple Measurements, one per day.
// This means that summing the measurements for a day is always <= 24 hours.
void AddMeasurementsFromDurationOrDie(int64_t start_time, int64_t end_time,
                                      const string& group, Series* series);

string LocationToGroup(const string& location);

// Process a line from a LifeCycle CSV file and add Measurement(s) to Series
// map. Each line produces one or more Measurements.
void ProcessLineOrDie(
    const std::string& line, const int64_t line_num,
    const boost::filesystem::path csv_path, SeriesCollection* const proto,
    absl::flat_hash_map<string, Series*>* const name_to_series_map);

// Process a SeriesCollection. The input message
// Any errors will lead to fatal program crash.
void ProcessSeriesCollectionOrDie(SeriesCollection* proto);

}  // namespace me
