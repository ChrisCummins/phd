// Protocol buffer processing binary for extracting me.Series protos from a
// YNAB JSON file.
#include "phd/logging.h"
#include "phd/pbutil.h"
#include "phd/string.h"

#include "datasets/me_db/me.pb.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/container/flat_hash_map.h"

#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace me {

template<typename K, typename V>
void InsertOrDie(absl::flat_hash_map<K, V>* map, const K& key, const V& value) {
  auto it = map->find(key);

  if (it == map->end()) {
    map->insert(std::make_pair(key, value));
  } else {
    FATAL("Duplicate key in map: %s", key);
  }
}

template<typename K, typename V>
V FindOrDie(const absl::flat_hash_map<K, V>& map, const K& key) {
  auto it = map.find(key);

  if (it == map.end()) {
    FATAL("Key not found in map: %s", key);
  } else {
    return it->second;
  }
}

int64_t ParseDateOrDie(const string& date) {
  absl::Time time;
  std::string err;
  bool succeeded = absl::ParseTime("%Y-%m-%d", date, &time, &err);
  if (!succeeded) {
    FATAL("Failed to parse '%s': %s", date, err);
  }
  absl::Duration d = time - absl::UnixEpoch();
  return d / absl::Milliseconds(1);
}

bool TryGetCategory(
    const string& category_id,
    const absl::flat_hash_map<string, string>& category_id_to_name,
    string* category) {
  if (category_id == "Category/__ImmediateIncome__") {
    *category = "Income";
  } else if (category_id == "Category/__DeferredIncome__") {
    *category = "Income";
  } else if (category_id == "null") {
    return false;
  } else {
    *category = FindOrDie(category_id_to_name, category_id);
  }
  return true;
}

// Extract and format the name for a budget from a filesystem path.
string GetBudgetNameFromPathOrDie(const boost::filesystem::path& path) {
  // Path has the format: .../<budget-dir>/<stuff>/<stuff>/Budget.yfull.
  // Begin by isolating <budget-dir> from the full path.
  CHECK(path.filename() == "Budget.yfull");
  const auto directory_name = (
      path.parent_path().parent_path().parent_path().filename().string());

  LOG(DEBUG) << "Directory: " << directory_name;
  CHECK(phd::EndsWith(directory_name, ".ynab4"));

  // <budget-dir> has the format '$NAME~<stuff>.ynab4'. Split at the ~ and
  // return the first component.
  std::vector<absl::string_view> components = absl::StrSplit(
      directory_name, '~');
  CHECK(components.size() == 2);

  const string unformatted_budget_name = string(components[0]);
  CHECK(!unformatted_budget_name.empty());

  return phd::ToCamelCase(unformatted_budget_name);
}

void TryAddTransactionMeasurementToSeries(
    const boost::property_tree::ptree& transaction, const int64_t date,
    const absl::flat_hash_map<string, string>& category_id_to_name,
    Series* series, string* category) {
  const string category_id = transaction.get<string>("categoryId", "");
  CHECK(!category_id.empty());

  if (!TryGetCategory(category_id, category_id_to_name, category)) {
    WARN("Failed to get category for id `%s`", category_id);
    return;
  }

  const float amount = transaction.get<double>("amount", 0);
  if (amount == 0.0) {
    WARN("Ignoring transaction with zero amount: %s at %d", *category, date);
    return;
  }

  Measurement* measurement = series->add_measurement();
  measurement->set_ms_since_unix_epoch(date);
  measurement->set_value(static_cast<int64_t>(amount * 100));
  measurement->set_group(*category);
  measurement->set_source("YNAB");
}

Series CreateTransactionsSeries(
    const string& budget_name,
    const absl::flat_hash_map<string, string>& category_id_to_name,
    const boost::property_tree::ptree& root) {
  Series series;
  series.set_name(absl::StrFormat("%sTransactions", budget_name));
  series.set_family("Finances");
  series.set_unit("pound_sterling_pence");

  string category;
  for (const boost::property_tree::ptree::value_type& transaction_elem :
       root.get_child("transactions")) {
    auto transaction = transaction_elem.second;

    const int64_t date = ParseDateOrDie(
        transaction.get<string>("date", "__no_date__"));

    const string category_id = transaction.get<string>("categoryId", "");

    if (category_id == "Category/__Split__") {
       for (const boost::property_tree::ptree::value_type& subtransaction_elem :
            transaction.get_child("subTransactions")) {
         auto subtransaction = subtransaction_elem.second;

         TryAddTransactionMeasurementToSeries(
            subtransaction, date, category_id_to_name, &series, &category);
       }
    } else {
      TryAddTransactionMeasurementToSeries(
          transaction, date, category_id_to_name, &series, &category);
    }
  }

  return series;
}

Series CreateBudgetSeries(
    const string& budget_name,
    const absl::flat_hash_map<string, string>& category_id_to_name,
    const boost::property_tree::ptree& root) {
  Series series;
  series.set_name(absl::StrFormat("%sBudget", budget_name));
  series.set_family("Finances");
  series.set_unit("pound_sterling_pence");

  string category;
  for (const boost::property_tree::ptree::value_type& month_elem :
       root.get_child("monthlyBudgets")) {
    auto month = month_elem.second;

    const int64_t date = ParseDateOrDie(
        month.get<string>("month", "__no_date__"));

    for (const boost::property_tree::ptree::value_type& budget_elem :
         month.get_child("monthlySubCategoryBudgets")) {
      auto budget = budget_elem.second;

      const string category_id = budget.get<string>("categoryId", "");
      CHECK(!category_id.empty());

      if (!TryGetCategory(category_id, category_id_to_name, &category)) {
        WARN("Failed to get category for id `%s`", category_id);
        continue;
      }

      const float amount = budget.get<double>("amount", 0);
      CHECK(amount >= 0.0);  // No negative budgets.

      Measurement* measurement = series.add_measurement();
      measurement->set_ms_since_unix_epoch(date);
      measurement->set_value(static_cast<int64_t>(amount * 100));
      measurement->set_group(category);
      measurement->set_source("YNAB");
    }
  }

  return series;
}

void ProcessJsonBudgetFile(SeriesCollection* proto) {
  const boost::filesystem::path json_path(proto->source());

  const string budget_name = GetBudgetNameFromPathOrDie(json_path);

  CHECK(boost::filesystem::is_regular_file(json_path));
  LOG(INFO) << "Reading from JSON file " << json_path.string();

  boost::filesystem::ifstream json(json_path);
  CHECK(json.is_open());

  boost::property_tree::ptree root;
  boost::property_tree::read_json(json, root);

  // Create a map from category IDs to category names.
  absl::flat_hash_map<string, string> category_id_to_name;
  for (const boost::property_tree::ptree::value_type& master_category_elem :
       root.get_child("masterCategories")) {
    auto master_category = master_category_elem.second;

    // Sanity check that master category is outflow.
    CHECK(master_category.get<string>("type", "") == "OUTFLOW");

    const string master_category_name = master_category.get<string>("name", "");
    CHECK(!master_category_name.empty());

    for (const boost::property_tree::ptree::value_type& subcategory_elem :
         master_category.get_child("subCategories")) {
      auto subcategory = subcategory_elem.second;

      const string subcategory_name = subcategory.get<string>("name", "");
      CHECK(!subcategory_name.empty());
      CHECK(subcategory.get<string>("type", "") == "OUTFLOW");

      // The category name is a concatenation of the master and subcategory
      // names.
      const string category_name = absl::StrFormat(
          "%s:%s", phd::ToCamelCase(master_category_name),
          phd::ToCamelCase(subcategory_name));

      const string subcategory_id = subcategory.get<string>("entityId", "");
      CHECK(!subcategory_id.empty());

      InsertOrDie(&category_id_to_name, subcategory_id, category_name);
    }
  }

  *proto->add_series() = CreateTransactionsSeries(budget_name, category_id_to_name, root);
  *proto->add_series() = CreateBudgetSeries(budget_name, category_id_to_name, root);
}

}  // namespace me

PBUTIL_INPLACE_PROCESS_MAIN(me::ProcessJsonBudgetFile, me::SeriesCollection);
