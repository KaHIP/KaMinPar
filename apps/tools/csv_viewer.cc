#include <iostream>
#include <sstream>
#include <vector>

const std::vector<std::string> colors = {
    "\u001b[31;1m", "\u001b[32;1m", "\u001b[33;1m", "\u001b[35;1m", "\u001b[36;1m", "\u001b[37;1m",
};

const std::string reset = "\u001b[0m";

std::vector<std::string> parse_row(const std::string &row) {
  std::stringstream ss(row);

  std::vector<std::string> columns;
  while (ss.good()) {
    std::string column;
    std::getline(ss, column, ',');
    columns.push_back(column);
  }

  return columns;
}

std::vector<bool> create_column_filter(const std::vector<std::string> &all_columns,
                                       const std::vector<std::string> &desired_columns) {
  std::vector<bool> filter;
  filter.reserve(all_columns.size());
  for (const std::string &column : all_columns) {
    filter.push_back(std::find(desired_columns.begin(), desired_columns.end(), column) < desired_columns.end());
  }
  return filter;
}

void print_row(const std::vector<std::string> &row, const std::vector<bool> &filter) {
  std::size_t c = 0;
  bool first = true;
  for (std::size_t i = 0; i < row.size(); ++i) {
    if (i >= filter.size() || !filter[i]) { continue; }
    if (!first) {
      std::cout << ",";
    } else {
      first = false;
    }
    std::cout << colors[c] << row[i] << reset;
    c = (c + 1) % colors.size();
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  std::vector<std::string> desired_columns;
  for (int i = 1; i < argc; ++i) { desired_columns.emplace_back(argv[i]); }

  std::string row;
  std::getline(std::cin, row);
  auto header = parse_row(row);

  std::vector<bool> filter;
  if (desired_columns.empty()) {
    filter.resize(header.size(), true);
  } else {
    filter = create_column_filter(header, desired_columns);
  }
  print_row(header, filter);

  while (std::getline(std::cin, row)) {
    auto columns = parse_row(row);
    print_row(columns, filter);
  }

  return 0;
}