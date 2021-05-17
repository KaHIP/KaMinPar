#include "csv.h"

#include "utility/utility.h"

namespace kaminpar::tool {
constexpr std::array colors = {
    "\u001b[31;1m"sv, "\u001b[32;1m"sv, "\u001b[33;1m"sv, "\u001b[35;1m"sv, "\u001b[36;1m"sv, "\u001b[37;1m"sv,
};
constexpr auto reset = "\u001b[0m"sv;

std::vector<std::size_t> find_column_indices(const Csv &csv, const std::vector<std::string> &columns) {
  std::vector<std::size_t> permutation(columns.size());

  for (std::size_t i = 0; i < columns.size(); ++i) {
    bool found_column{false};
    const std::string name = utility::str::to_lower(columns[i]);

    for (std::size_t j = 0; j < csv.columns.size(); ++j) {
      const std::string current_name = utility::str::to_lower(csv.columns[j]);
      if (name == current_name) {
        permutation[i] = j;
        found_column = true;
        break;
      }
    }
    if (!found_column) { FATAL_ERROR << "column " << columns[i] << " not found in on of the csv files"; }
  }

  return permutation;
}

Csv merge_csvs(const std::vector<std::string> &common_columns, const std::vector<Csv> &csvs) {
  Csv result{};
  result.columns = common_columns;

  for (const auto &csv : csvs) {
    auto permutation = find_column_indices(csv, common_columns);
    for (const auto &current_row : csv.rows) {
      std::vector<std::string> filtered_row{};
      for (const std::size_t &i : permutation) { filtered_row.push_back(current_row[i]); }
      result.rows.push_back(filtered_row);
    }
  }

  return result;
}

double aggregate(const std::string &function, const std::vector<double> &values) {
  if (function == "count") {
    return values.size();
  } else if (function == "avg") {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  } else if (function == "max") {
    return *std::max_element(values.begin(), values.end());
  } else if (function == "min") {
    return *std::min_element(values.begin(), values.end());
  } else if (function == "sum") {
    return std::accumulate(values.begin(), values.end(), 0.0);
  } else {
    FATAL_ERROR << "unknown aggregate function: " << function;
    return 0.0;
  }
}

Csv aggregate(const Csv &csv, const std::vector<std::string> &group_by,
              const std::vector<std::pair<std::string, std::string>> &aggregates,
              const std::size_t expected_rows_per_aggregate) {
  Csv result{};
  result.columns = csv.columns;

  auto group_by_indices = find_column_indices(csv, group_by);
  std::vector<std::string> aggregate_columns;
  for (const auto &[column, function] : aggregates) { aggregate_columns.push_back(column); }
  auto aggregate_indices = find_column_indices(csv, aggregate_columns);

  std::unordered_map<std::string, std::vector<std::vector<double>>> data;
  for (const auto &row : csv.rows) {
    std::stringstream key_ss;
    bool first{true};
    for (const std::size_t i : group_by_indices) {
      if (first) {
        first = false;
      } else {
        key_ss << ",";
      }
      key_ss << row[i];
    }
    const std::string key = key_ss.str();

    if (!data.contains(key)) { data[key].resize(aggregate_columns.size()); }

    for (std::size_t i = 0; i < aggregate_indices.size(); ++i) {
      data[key][i].push_back(std::strtod(row[aggregate_indices[i]].c_str(), nullptr));
    }
  }

  for (const auto &[key, entries] : data) {
    std::vector<std::string> row(result.columns.size());
    auto key_parts = utility::str::explode(key, ',');
    for (std::size_t i = 0; i < group_by_indices.size(); ++i) { row[group_by_indices[i]] = key_parts[i]; }

    if (!aggregate_indices.empty()) {
      const auto &values = entries[0];
      if (expected_rows_per_aggregate != 0 && values.size() != expected_rows_per_aggregate) {
        WARNING << "Expected " << expected_rows_per_aggregate << " rows to aggregate for key " << key
                << ", but only got " << values.size();
      }
    }

    for (std::size_t i = 0; i < aggregate_indices.size(); ++i) {
      const auto &values = entries[i];
      const double value = aggregate(aggregates[i].second, values);
      row[aggregate_indices[i]] = std::to_string(value);
    }
    result.rows.push_back(std::move(row));
  }
  return result;
}

Csv load_csv(const std::string &filename, const bool filter_failed, const char del, const bool quiet) {
  std::ifstream in(filename);
  if (!in) { FATAL_ERROR << "Cannot read " << filename; }

  Csv csv{};

  { // parse header
    std::string header;
    std::getline(in, header);
    csv.columns = utility::str::explode(header, del);
  }

  std::size_t failed_index = 0;
  for (std::size_t i = 0; i < csv.columns.size(); ++i) {
    if (csv.columns[i] == "Failed") {
      failed_index = i;
      break;
    }
  }

  std::size_t filtered_rows = 0;

  { // parse rows
    std::string line;
    while (std::getline(in, line)) {
      auto row = utility::str::explode(line, del);
      if (filter_failed && row[failed_index] == "1") {
        ++filtered_rows;
        continue;
      }
      csv.rows.push_back(std::move(row));
      if (csv.rows.back().size() != csv.columns.size()) {
        FATAL_ERROR << "Row has not enough columns:\n" << line << "\n(expected " << csv.columns.size() << " columns)";
      }
    }
  }

  if (!quiet && filtered_rows > 0) {
    WARNING << "While loading " << filename << ": removed " << filtered_rows << " failed runs";
  }
  return csv;
}

void print_row(const std::vector<std::string> &row, const bool colorize, const char del) {
  for (std::size_t i = 0; i < row.size(); ++i) {
    if (colorize) { std::cout << colors[i % colors.size()]; }
    if (i > 0) { std::cout << del; }
    std::cout << row[i];
    if (colorize) { std::cout << reset; }
  }
}

void print_csv(const Csv &csv, const bool colorize, const char del) {
  print_row(csv.columns, colorize, del);
  std::cout << "\n";
  for (const auto &row : csv.rows) {
    print_row(row, colorize, del);
    std::cout << "\n";
  }
  std::cout << std::flush;
}
} // namespace kaminpar::tool